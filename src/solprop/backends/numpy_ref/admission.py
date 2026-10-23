# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route's and the coupled flow's admission, for every backend.

One list, run by each backend under its own name: ``numpy_ref`` calls
:func:`refuse_general_route` from its own admission, the native backend calls
it from its adapter, and ``numba_cpu`` runs the same checks. A combination admitted or
refused here is admitted or refused on every backend that calls it, and each refusal
names the calling backend and the leaf, so a new refusal is written once and cannot
drift between copies.

What a backend has not built is said with `flow_methods`: the section methods of
``linear.coupled_flow`` it executes. A section method outside them is refused by name
before anything is allocated. Everything else -- the batch axis, a backend's own
kernels' limits -- stays with the backend that has the limit.

The list runs **before** :meth:`solprop.translate.problem.KernelProblem.validate` and
before any buffer is allocated, and repeats checks the translation layer also makes,
because a problem can be assembled by something other than
:func:`solprop.translate.problem.assemble` and a backend does not rely on a caller
having been careful.
"""

from collections.abc import Collection
from typing import Final

import numpy as np

from solprop.backends.numpy_ref.general_modal import refuse_regional_responses
from solprop.exceptions import BackendError
from solprop.translate.coupled import (
    CoupledFlowTables,
    ModalMetricTables,
    SectionMethod,
)
from solprop.translate.general import MAX_DENSE_METRIC_DEPARTURE, metric_departure
from solprop.translate.linop import LinearOperator
from solprop.translate.modal_tables import DEFAULT_GRAM_ATOL, ModalTransformTables
from solprop.translate.problem import MAX_N_COMPONENTS, KernelProblem

__all__ = ["FLOW_METHODS", "refuse_coupled_flow", "refuse_general_route"]

FLOW_METHODS: Final[tuple[SectionMethod, ...]] = (
    "constant_schur",
    "frozen_residual",
    "cf4",
)
"""Every section method of ``linear.coupled_flow`` the reference executes.

``"diagonal"`` is not among them: it is the existing cumulative flow and needs no
leaf. A backend that executes fewer passes its own subset to
:func:`refuse_coupled_flow`; ``"cf4"`` is executed by the reference alone: it is not yet
implemented on the compiled backends.
"""


def refuse_general_route(
    problem: KernelProblem,
    *,
    backend: str = "numpy_ref",
    flow_methods: Collection[SectionMethod] = FLOW_METHODS,
    moving_maps: bool = False,
) -> None:
    """Refuse, by name, every general modal or coupled-flow problem not executed.

    The coupled flow first: a metric or a table beside it is this route's refusal to
    make, not the general route's, and its message names the flow
    (:func:`refuse_coupled_flow`). Then the general modal route: a field grid, one
    channel axis and no state polarisation axis, the tables' own execution maps of a
    constant guide, an identity metric or the dense metric of a declared unitary change
    of basis of one, the constant unit power weight, no legacy response beside the
    route's own, and per-region responses the route executes
    (``refuse_regional_responses``, in :mod:`solprop.backends.numpy_ref.general_modal`).
    A problem carrying none of the four general-modal leaves passes untouched.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem to check.
    backend : str, optional
        The registry name each refusal names.
    flow_methods : collection of str, optional
        The section methods of ``linear.coupled_flow`` the backend executes; every
        one of :data:`FLOW_METHODS` by default, which is the reference's own list.
    moving_maps : bool, optional
        Whether the backend executes a **moving** guide's maps -- synthesis and
        projection of more than one coefficient along z, combined per stage over
        the bundle's ``transform_weights`` (kernel schema 14). False by default, so
        a backend that has not built them refuses them by name whatever else it
        admits; ``numpy_ref``, ``numba_cpu`` and the native backend pass True.

    Raises
    ------
    BackendError
        Naming the calling backend and what it does not execute.

    Examples
    --------
    >>> from solprop.backends.numpy_ref.admission import refuse_general_route
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
    >>> refuse_general_route(problem, backend="native") is None
    True
    """
    refuse_coupled_flow(problem, backend=backend, flow_methods=flow_methods)
    _refuse_modal_route(problem, backend=backend, moving_maps=moving_maps)


def refuse_coupled_flow(
    problem: KernelProblem,
    *,
    backend: str = "numpy_ref",
    flow_methods: Collection[SectionMethod] = FLOW_METHODS,
) -> None:
    """Refuse, by name, every coupled flow the calling backend does not execute.

    The reference executes exactly two combinations: the dense generator of **one**
    constant section with no connection and no residual, by a Schur matrix-function
    action, and a **varying** flow whose every section is ``"frozen_residual"`` or
    ``"cf4"`` -- the dense generator at any degree along z and an optional connection,
    a frozen-residual section by the Schur action of its frozen block with a residual
    the interaction-picture source carries explicitly, and a CF4 section by the
    commutator-free exponentials of its whole generator with none. A backend's
    `flow_methods` narrows that; naming
    ``linear.coupled_flow`` among a backend's admitted leaves says only that the field
    may be present, and this says which contents are executed: every other method,
    representation, mixture of the two and certificate row is named here rather than
    silently ignored.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem to check.
    backend : str, optional
        The registry name each refusal names.
    flow_methods : collection of str, optional
        The section methods the backend executes (:data:`FLOW_METHODS` by default).

    Raises
    ------
    BackendError
        Naming the part of the leaf, or of the problem around it, it cannot execute.
    """
    flow = problem.linear.coupled_flow
    if flow is None:
        return
    _refuse_flow_representation(flow, backend=backend, flow_methods=flow_methods)
    _refuse_flow_problem(problem, backend=backend)


def _refuse_flow_representation(
    flow: CoupledFlowTables,
    *,
    backend: str,
    flow_methods: Collection[SectionMethod],
) -> None:
    """Refuse a coupled flow whose representation the backend does not execute.

    Parameters
    ----------
    flow : solprop.translate.coupled.CoupledFlowTables
        The leaf.
    backend : str
        The registry name each refusal names.
    flow_methods : collection of str
        The section methods the backend executes.

    Raises
    ------
    BackendError
        Naming the method, the representation, the section count or the connection.
    """
    methods = set(flow.section_methods)
    frozen = methods <= {"frozen_residual", "cf4"}
    if flow.section_methods != ("constant_schur",) and not frozen:
        varying = " or ".join(
            repr(method) for method in flow_methods if method != "constant_schur"
        )
        msg = (
            f"{backend} executes one constant section by a Schur matrix-function "
            f"action, or every section of a varying flow as {varying}, and this "
            f"flow declares {sorted(methods)} over "
            f"{len(flow.section_methods)} section(s): 'diagonal' is the existing "
            "cumulative flow and needs no leaf, and a constant-Schur section beside "
            "any other is not one this backend builds a single held decomposition for"
        )
        raise BackendError(msg)
    unexecuted = sorted(methods - set(flow_methods))
    if unexecuted:
        msg = (
            f"{backend} does not execute a {' or '.join(map(repr, unexecuted))} "
            "section of linear.coupled_flow: it executes "
            f"{', '.join(map(repr, sorted(flow_methods)))} sections, and a section "
            "method silently integrated by another would run, converge and be a "
            "different approximation. The commutator-free CF4 flow is not yet "
            "implemented on the compiled backends; pass backend='numpy_ref', or "
            "declare the default frozen split"
        )
        raise BackendError(msg)
    if flow.representation != "dense":
        msg = (
            f"{backend} executes the dense generator of a coupled flow and this one "
            f"is {flow.representation!r}: a factored generator is reconstructed by no "
            "kernel here"
        )
        raise BackendError(msg)
    assert flow.generator is not None
    if not frozen and flow.generator.shape[1] != 1:
        msg = (
            f"{backend} executes a constant-Schur coupled flow whose generator is "
            f"constant across its section, and this one carries "
            f"{flow.generator.shape[1]} Chebyshev degrees along z: a generator that "
            "varies inside its section is a 'frozen_residual' flow, which declares "
            "the residual its variation becomes"
        )
        raise BackendError(msg)
    if not frozen and flow.connection is not None:
        msg = (
            f"{backend} executes no taper connection beside a constant section: "
            "linear.coupled_flow.connection is the moving-basis term of a taper and "
            "belongs to a 'frozen_residual' flow, whose explicit residual carries it"
        )
        raise BackendError(msg)
    if not frozen and float(np.max(flow.residual_norm_bound)) > 0.0:
        msg = (
            f"{backend} executes a constant-Schur coupled flow with no explicit "
            f"residual, and this one declares up to "
            f"{float(np.max(flow.residual_norm_bound)):.4g} 1/m: a section whose "
            "generator varies inside it is a 'frozen_residual' flow"
        )
        raise BackendError(msg)


def _refuse_flow_problem(problem: KernelProblem, *, backend: str) -> None:
    """Refuse a problem whose surroundings the coupled flow cannot be executed in.

    The four that are this route's to make. The cumulative diagonal ``Phi``
    multiplies the flow action, which is only the same equation when the two
    **commute**, and a mode-dependent diagonal does not, so ``phi_constant`` is
    required to be a multiple of the identity; a shot-noise draw is transported by the
    unitary phase of a diagonal flow, which a nonunitary coupled flow is
    not; the flux of a coupled block is the square-root model's, whose positivity
    is not established, so no explicit metric may sit beside one; and the block is
    **scalar**, so the state carries one polarisation.

    That last one is refused in the translation layer too
    (:func:`solprop.translate.problem._check_wall_route`, which owns the mode set the
    block is built from). It is repeated here because a problem can be assembled by
    something other than :func:`solprop.translate.problem.assemble`, and because the
    action contracts the channel axis alone: a scalar block beside two state
    components would be applied to each of them as though it described both. The TE/TM
    wall block (``CoherentWall(impedance="te_tm")``) names
    the wall's two impedances and is scalar in this sense too: its families are
    channels. The **transform** may resolve two transverse components (a ``TE_0m``,
    ``TM_0m`` or ``EH`` set synthesises E_x and E_y): the synthesis
    is read by the responses alone, never by the flow, so it is not refused.

    A batch axis is refused for every problem by each backend's own ``max_batch = 1``
    before anything reaches here, and is **not** repeated.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem, whose coupled flow is not ``None``.
    backend : str
        The registry name each refusal names.

    Raises
    ------
    BackendError
        Naming what the problem carries beside the flow.
    """
    _, _, n_pol, _ = problem.shape
    if n_pol != 1:
        msg = (
            f"{backend} executes a coupled flow only for a scalar state, and this "
            f"problem's state carries n_pol = {n_pol} polarisation components (the "
            "product route): the flow's action contracts the channel axis alone, so "
            "a scalar block beside a state with two polarisation components would be "
            "applied to each as though it described both. "
            "CoherentWall(impedance='te_tm') names the wall's TE and TM impedances, "
            "not the state's polarisation axis: its families are channels, one "
            "component each"
        )
        raise BackendError(msg)
    linear = problem.linear
    flow = linear.coupled_flow
    assert flow is not None
    constant = flow.section_methods == ("constant_schur",)
    if constant and (
        linear.phi_representation != "constant" or linear.phi_constant is None
    ):
        msg = (
            f"{backend} executes a constant-Schur coupled flow only beside the exact "
            f"constant representation of Phi, and this problem's is "
            f"{linear.phi_representation!r}: a varying carrier belongs to a "
            "'frozen_residual' flow"
        )
        raise BackendError(msg)
    if not _mode_independent(linear):
        msg = (
            f"{backend} executes a coupled flow only beside a mode-independent Phi: "
            "the flow action and the cumulative diagonal are applied as a product, "
            "which is the same equation only when they commute, and a channel-"
            "dependent diagonal does not commute with a coupled block"
        )
        raise BackendError(msg)
    if not constant and not problem.device_ready:
        msg = (
            f"{backend} executes a frozen-residual coupled flow only beside a Phi "
            "that is a weighted sum of fixed terms, and this problem's took the z "
            "table: the flow's own stage weights and the carrier's must come from "
            "one stage bundle, and the table has no such form "
            "(docs/dev/kernel-contract.md section 1)"
        )
        raise BackendError(msg)
    if problem.initial.noise0 is not None:
        msg = (
            f"{backend} executes no noise model beside a coupled flow: the "
            "one-photon-per-bin draw is advanced by the unitary phase of a diagonal "
            "flow, and its transport through a nonunitary coupled flow is "
            "not measured in this package"
        )
        raise BackendError(msg)
    if linear.modal_metric is not None and problem.transform.modal_tables is None:
        # The refusal is about the **coherent wall's** own flux model: a condensed
        # boundary block has no bound reciprocal basis, so its flux is the square-root
        # model's and its positivity is not established. A general modal guide's metric
        # is a different object -- the Gram of its own sampled, power-orthonormal
        # transverse basis, which the route measures and gates before it declares
        # anything -- and the coupled block there is a declared unitary rotation of that
        # basis, which leaves the Gram's spectrum exactly where it was.
        # `transform.modal_tables` is what tells the two apart, because a mode-set route
        # has none and never will.
        msg = (
            f"{backend} executes no explicit metric beside a coupled flow on the "
            "mode-set route: the flux of a condensed boundary block is the "
            "square-root model's, whose positivity is not established. A general "
            "modal guide carries "
            "transform.modal_tables and its metric is its own sampled basis Gram, "
            "which this problem does not"
        )
        raise BackendError(msg)


def _mode_independent(linear: LinearOperator) -> bool:
    """Return whether the cumulative ``Phi`` is a multiple of the identity in channels.

    That, and not constancy, is what the product ``exp(dPhi) exp(dz G_s)`` needs: a
    scalar multiple of the identity commutes with any block, a channel-dependent
    diagonal does not. Every representation is checked in its own terms -- the exact
    constant array, the expansion's fixed terms, or the table's coefficients -- because
    a varying guide takes whichever of them its declared parameters allow.

    Parameters
    ----------
    linear : solprop.translate.linop.LinearOperator
        Section 3 of the contract.

    Returns
    -------
    bool
        True when every channel of every term carries the same values.
    """
    held: list[np.ndarray] = []
    if linear.phi_constant is not None:
        held.append(linear.phi_constant)
    if linear.phi_expansion is not None:
        held.append(linear.phi_expansion.terms)
    if linear.phi_coefficients is not None:
        held.append(linear.phi_coefficients)
    return all(
        np.array_equal(
            block,
            np.broadcast_to(np.take(block, [0], axis=-3), block.shape),
        )
        for block in held
    )


def _refuse_general_metric(
    tables: ModalTransformTables,
    metric: ModalMetricTables,
    *,
    backend: str,
) -> None:
    """Refuse a dense metric this backend cannot execute, measuring it itself.

    The **nonunitary refusal**, at the backend. ``numpy_ref`` executes a general
    modal guide through the **diagonal** cumulative flow and measures the step
    controller's norm on the coordinate amplitudes, both of which are the basis's
    own only where that basis is orthonormal in true power. A declared unitary
    change of basis of a certified orthonormal reference stays there --
    ``W' = I + U.H delta U`` has the same spectral departure as ``delta`` -- and a
    nonunitary one does not: it is a skewed or leaky basis, whose flow is
    ``linear.coupled_flow`` and whose transport nothing here has measured.

    The block is measured here rather than read off ``error``: the translation layer
    declares that row, and a backend does not rely on a caller having been careful
    (the same rule the rest of this admission path follows).

    Parameters
    ----------
    tables : solprop.translate.modal_tables.ModalTransformTables
        The reference the metric must describe.
    metric : solprop.translate.coupled.ModalMetricTables
        The admitted identity or dense metric.
    backend : str
        The registry name the refusal names.

    Raises
    ------
    BackendError
        If a dense metric is on another basis than the reference, or is further from
        orthonormal than ``numpy_ref`` will call the identity.
    """
    block = metric.dense
    if block is None:
        return
    if metric.gauge_id != tables.gauge_id:
        msg = (
            f"linear.modal_metric describes the basis {metric.gauge_id!r} and "
            f"transform.modal_tables is in {tables.gauge_id!r}: a metric of another "
            "basis than the reference it is executed beside would weight the wrong "
            "channels"
        )
        raise BackendError(msg)
    departure = max(
        metric_departure(block[interval, degree])
        for interval in range(block.shape[0])
        for degree in range(block.shape[1])
    )
    if departure > MAX_DENSE_METRIC_DEPARTURE:
        msg = (
            f"linear.modal_metric is dense and {departure:.4g} from the identity, "
            f"above the {MAX_DENSE_METRIC_DEPARTURE:.4g} {backend} will execute. "
            "This backend propagates through the diagonal cumulative flow and "
            "measures the controller's norm on the coordinate amplitudes, so it "
            "admits a dense metric only as a unitary change of basis of a "
            "certified orthonormal reference, whose departure is the reference's "
            "own; a nonunitary transform leaves a genuinely nonorthogonal basis, "
            "whose flow is linear.coupled_flow and whose transport is unmeasured"
        )
        raise BackendError(msg)


def _refuse_modal_route(
    problem: KernelProblem, *, backend: str, moving_maps: bool = False
) -> None:
    """Refuse every general modal problem the calling backend does not execute.

    The general modal route is one combination: a field grid, one channel axis and no
    state polarisation axis, the tables' own execution maps, an **identity** metric --
    the Poynting-normalised orthogonal basis the Gram gate certifies -- the constant
    unit power weight of a true-power amplitude, three Cartesian components in the
    responses as in the reference, and constant geometry.
    Everything else is refused by name, because a representation silently ignored
    would run, converge and be the wrong physics.

    Two of these are also checked in the translation layer, where the record is
    built: the state's ``n_pol`` and the metric's declared defect. They are checked
    again here because this list runs *before* ``KernelProblem.validate`` and because
    a problem can be assembled by something other than
    :func:`solprop.translate.problem.assemble`; a backend does not rely on a caller
    having been careful. Every refusal is raised before the route's maps, tensors or
    driver buffers are allocated.

    A legacy response beside the route's own is refused too: the route's source
    replaces the node pipeline that reads ``kerr_coefficients``, the population
    channels, the Raman terms and the spectral responses, so any of them would be
    carried in the record and silently not propagated. Every backend refuses it through
    this one list.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem to check.
    backend : str
        The registry name each refusal names.
    moving_maps : bool, optional
        Whether the backend executes a moving guide's maps
        (:func:`refuse_general_route`).

    Raises
    ------
    BackendError
        Naming what is unsupported.
    """
    tables = problem.transform.modal_tables
    metric = problem.linear.modal_metric
    if tables is None:
        for name, leaf in (
            ("linear.modal_metric", metric),
            ("responses.regional_responses", problem.responses.regional_responses),
        ):
            if leaf is not None:
                msg = (
                    f"{backend} executes {name} as part of the general modal route, "
                    "and this problem carries no transform.modal_tables for it to "
                    "belong to"
                )
                raise BackendError(msg)
        return
    if metric is None or metric.kind not in ("identity", "dense"):
        kind = "absent" if metric is None else f"of kind {metric.kind!r}"
        msg = (
            f"{backend} executes a general modal guide in a basis certified "
            f"orthonormal in true power -- an identity linear.modal_metric, or the "
            f"dense metric of a declared unitary change of basis of one -- and this "
            f"problem's is {kind}; a diagonal metric is a power-orthogonal basis "
            "with unequal channel norms, which nothing in this package builds and "
            "no backend has measured"
        )
        raise BackendError(msg)
    defect = float(np.max(metric.error))
    if defect > DEFAULT_GRAM_ATOL:
        # The translation layer gates this at the route's own declared budget; a
        # certificate that reaches a backend ungated is an unbounded silent error,
        # because an identity metric is executed as the identity and nothing
        # downstream reads the error row again.
        msg = (
            f"linear.modal_metric.error is {defect:.4g}, above the "
            f"{DEFAULT_GRAM_ATOL:.4g} {backend} will call the identity: this basis "
            "is that far from orthonormal where it was sampled, and a continuous "
            "wave's da/dz would be wrong by about that much"
        )
        raise BackendError(msg)
    _refuse_general_metric(tables, metric, backend=backend)
    if problem.grid_kind != "field":
        msg = (
            f"{backend} executes a general modal guide on a field grid, whose "
            f"canonical axis is increasing, and this problem is {problem.grid_kind}"
        )
        raise BackendError(msg)
    if tables.synthesis is None or tables.projection is None:
        msg = (
            f"{backend} executes a general modal guide through the synthesis and "
            "projection maps of transform.modal_tables, and this problem carries "
            "neither: a reference sampled over a path of positions has no execution "
            "map"
        )
        raise BackendError(msg)
    if tables.synthesis.shape[0] != 1 or tables.synthesis.shape[1] != 1:
        _refuse_moving_maps(problem, tables, backend=backend, moving_maps=moving_maps)
    weight = problem.linear.power_weight_constant
    if weight is None or not np.all(weight == 1.0):
        msg = (
            f"{backend} executes a general modal guide whose Poynting amplitude is "
            "already the true-power amplitude, so linear.power_weight is the "
            "constant one; this problem's is "
            f"{problem.linear.power_weight_representation}"
        )
        raise BackendError(msg)
    n_batch = problem.shape[0]
    if n_batch != 1:
        # A **fused** batch axis, which is not what a general-guide batch is. The
        # stage passes `lab[0]` to the source pass, so a fused axis would propagate
        # member zero's polarisation for every member and nothing downstream would
        # say so. A general-guide batch is instead a lockstep group of singleton
        # workspaces, one per member, each handed its own row by
        # `NumpyBatchBackend`, so
        # this refusal stands and is never reached on that path.
        msg = (
            f"{backend} executes a general modal guide one member at a time and "
            f"this problem carries a batch axis of {n_batch}: the source pass "
            "evaluates the polarisation of member zero only. A general-guide "
            "batch is a lockstep group of singleton members sharing one guide, "
            "which solprop.driver.batch builds; a fused batch axis is not "
            "implemented. Run each launch on its own"
        )
        raise BackendError(msg)
    n_pol = problem.shape[2]
    if n_pol != 1:
        # Without this the pass would return a one-wide source and `stage_output`
        # would broadcast it onto a two-wide prefactor: two polarisations driven by
        # one channel's polarisation, which is not this route's physics.
        msg = (
            f"{backend} executes a general modal guide whose {MAX_N_COMPONENTS} "
            "physical components are the reference's own, so the state keeps "
            f"n_pol = 1, and this problem carries {n_pol}; a general guide puts "
            "every physical channel, including a selected parity partner, in "
            "n_modes"
        )
        raise BackendError(msg)
    _refuse_legacy_responses(problem, backend=backend)
    refuse_regional_responses(
        tables, problem.responses.regional_responses, backend=backend
    )


def _refuse_moving_maps(
    problem: KernelProblem,
    tables: ModalTransformTables,
    *,
    backend: str,
    moving_maps: bool,
) -> None:
    """Refuse a moving guide's maps where they are not executed, or not composed.

    A backend that has not built the per-stage combination refuses them by name.
    ``numpy_ref`` and ``numba_cpu`` execute them, and only beside what a moving
    guide is propagated with: a coupled flow carrying the connection, integrated as
    ``"frozen_residual"`` on every section. Moving maps with no connection are a
    guide whose modes change shape and exchange no power, which is the adiabatic
    approximation the route exists to replace; and CF4 beside moving maps composes
    on paper -- its Gauss-node exponentials read ``flow_weights`` and the source
    reads ``transform_weights`` at stage positions -- but is not built or measured
    on this route, so it is refused rather than run.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem, whose maps carry more than one coefficient along z.
    tables : solprop.translate.modal_tables.ModalTransformTables
        Its modal tables.
    backend : str
        The registry name each refusal names.
    moving_maps : bool
        Whether the backend executes moving maps at all.

    Raises
    ------
    BackendError
        Naming what is not executed.
    """
    assert tables.synthesis is not None
    intervals, degrees = (int(one) for one in tables.synthesis.shape[:2])
    if not moving_maps:
        msg = (
            f"{backend} executes a general modal guide of constant geometry, and "
            f"transform.modal_tables carries moving maps -- {intervals} section(s) "
            f"of degree {degrees - 1} along z, combined per stage over the bundle's "
            "transform_weights (kernel schema 14), which it does not build: a moving "
            "solid guide runs on numpy_ref, numba_cpu and native"
        )
        raise BackendError(msg)
    flow = problem.linear.coupled_flow
    if (
        flow is None
        or flow.connection is None
        or set(flow.section_methods) != {"frozen_residual"}
    ):
        described = (
            "no coupled flow"
            if flow is None
            else f"a flow of {sorted(set(flow.section_methods))} sections"
            + ("" if flow.connection is not None else " and no connection")
        )
        msg = (
            f"{backend} executes a moving guide's maps beside the flow that carries "
            f"its connection, 'frozen_residual' on every section, and this problem "
            f"carries {described}: maps that move with no connection are the "
            "adiabatic approximation the moving route replaces, and CF4 beside "
            "moving maps is not built or measured on this route"
        )
        raise BackendError(msg)


def _refuse_legacy_responses(problem: KernelProblem, *, backend: str) -> None:
    """Refuse a legacy response a general modal guide would carry and not propagate.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A problem carrying ``transform.modal_tables``.
    backend : str
        The registry name the refusal names.

    Raises
    ------
    BackendError
        Naming every legacy response the problem carries.
    """
    responses = problem.responses
    legacy = [
        name
        for name, present in (
            ("kerr_coefficients", responses.kerr_coefficients is not None),
            # The general route's own dispersive channels name their node regions;
            # a population's leaf has none and would not be read.
            (
                "dispersive_kerr",
                responses.dispersive_kerr is not None
                and responses.dispersive_kerr.regions is None,
            ),
            ("population_channels", bool(responses.population_channels)),
            ("raman_terms", bool(responses.raman_terms)),
            ("envelope_plasma", responses.envelope_plasma is not None),
            ("spectral_responses", bool(responses.spectral_responses)),
        )
        if present
    ]
    if legacy:
        msg = (
            f"{backend} executes transform.modal_tables with its own "
            "responses.regional_responses and no legacy response, and this problem "
            f"also carries responses.{', responses.'.join(legacy)}: the general "
            "route's source replaces the node pipeline that would read them, so they "
            "would be recorded and not propagated"
        )
        raise BackendError(msg)

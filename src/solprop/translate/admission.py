# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the general modal route and the coupled flow execute, checked once.

Both backends execute the same combinations of these leaves, so the list is part of the
problem's validation (:meth:`solprop.translate.problem.PropagationProblem.validate`,
which :func:`solprop.translate.problem.validate_run` runs where a backend is built):
each refusal has one raise site, before any buffer is allocated. It holds the policy:
the leaves' consistency with the state -- channel and component counts, axes, extents
-- is :meth:`~solprop.translate.problem.PropagationProblem.validate`'s own and runs
first. A problem can be assembled by something other than
:func:`solprop.translate.assembly.assemble`, so nothing here relies on the assembly
having refused it already.
"""

from typing import TYPE_CHECKING

import numpy as np

from solprop.exceptions import TranslationError
from solprop.translate.coupled import CoupledFlowTables, ModalMetricTables
from solprop.translate.general import (
    MAX_DENSE_METRIC_DEPARTURE,
    check_regional_responses,
    metric_departure,
)
from solprop.translate.linop import LinearOperator
from solprop.translate.modal_tables import DEFAULT_GRAM_ATOL, ModalTransformTables
from solprop.translate.responses import RegionalResponseTables

if TYPE_CHECKING:
    from solprop.translate.problem import PropagationProblem

__all__ = ["check_general_route", "check_leaf_types"]


def check_general_route(problem: "PropagationProblem") -> None:
    """Refuse, by name, each general modal or coupled-flow problem solprop does not run.

    The coupled flow first: a metric or a table beside it is this route's refusal to
    make, not the general route's. Then the general modal route: the tables' own
    execution maps, an identity metric or the dense metric of a declared unitary change
    of basis of one, the constant unit power weight, no legacy response beside the
    route's own, per-region responses of a known model, and moving maps only beside the
    flow that carries their connection. Its field grid is the leaves' own structure's to
    require: a leaf's increasing frequency axis is ``grids.omega``, which an envelope
    grid's FFT-ordered axis is not. A problem carrying none of the general-modal leaves
    passes untouched.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem to check, its leaves consistent with its state.

    Raises
    ------
    TranslationError
        Naming the leaf, or the part of the problem around it, that is not executed.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.translate.admission import check_general_route
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], wavelength=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    ...     width=100e-15, peak_power=9.8, shape="sech",
    ... )
    >>> check_general_route(simulation.translate()) is None
    True
    """
    flow = problem.linear.coupled_flow
    if flow is not None:
        _check_flow_representation(flow)
        _check_flow_problem(problem, flow)
    _check_modal_route(problem)


def check_leaf_types(problem: "PropagationProblem") -> None:
    """Refuse a general-modal leaf that is not its record type, by name.

    The first of the leaves' consistency checks
    (:meth:`~solprop.translate.problem.PropagationProblem.validate`), so that a
    hand-built problem holding something else there meets a named refusal rather than
    an attribute error further in.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem to check.

    Raises
    ------
    TranslationError
        Naming the leaf, the type it holds and the type it must be.
    """
    for name, leaf, kind in (
        (
            "transform.modal_tables",
            problem.transform.modal_tables,
            ModalTransformTables,
        ),
        ("linear.modal_metric", problem.linear.modal_metric, ModalMetricTables),
        ("linear.coupled_flow", problem.linear.coupled_flow, CoupledFlowTables),
        (
            "responses.regional_responses",
            problem.responses.regional_responses,
            RegionalResponseTables,
        ),
    ):
        if leaf is not None and not isinstance(leaf, kind):
            msg = f"{name} is a {type(leaf).__name__}, not a {kind.__name__}"
            raise TranslationError(msg)


def _check_flow_representation(flow: CoupledFlowTables) -> None:
    """Refuse a coupled flow whose representation no backend executes."""
    methods = set(flow.section_methods)
    frozen = methods == {"frozen_residual"}
    if flow.section_methods != ("constant_schur",) and not frozen:
        msg = (
            "solprop executes one constant section by a Schur matrix-function "
            "action, or every section of a varying flow as 'frozen_residual', and this "
            f"flow declares {sorted(methods)} over "
            f"{len(flow.section_methods)} section(s): 'diagonal' is the existing "
            "cumulative flow and needs no leaf, and a constant-Schur section beside "
            "any other is not one a single held decomposition is built for"
        )
        raise TranslationError(msg)
    if flow.representation != "dense":
        msg = (
            "solprop executes the dense generator of a coupled flow and this one "
            f"is {flow.representation!r}: a factored generator is reconstructed by no "
            "kernel"
        )
        raise TranslationError(msg)
    assert flow.generator is not None
    if not frozen and flow.generator.shape[1] != 1:
        msg = (
            "solprop executes a constant-Schur coupled flow whose generator is "
            f"constant across its section, and this one carries "
            f"{flow.generator.shape[1]} Chebyshev degrees along z: a generator that "
            "varies inside its section is a 'frozen_residual' flow, which declares "
            "the residual its variation becomes"
        )
        raise TranslationError(msg)
    if not frozen and flow.connection is not None:
        msg = (
            "solprop executes no taper connection beside a constant section: "
            "linear.coupled_flow.connection is the moving-basis term of a taper and "
            "belongs to a 'frozen_residual' flow, whose explicit residual carries it"
        )
        raise TranslationError(msg)
    if not frozen and float(np.max(flow.residual_norm_bound)) > 0.0:
        msg = (
            "solprop executes a constant-Schur coupled flow with no explicit "
            f"residual, and this one declares up to "
            f"{float(np.max(flow.residual_norm_bound)):.4g} 1/m: a section whose "
            "generator varies inside it is a 'frozen_residual' flow"
        )
        raise TranslationError(msg)


def _check_flow_problem(problem: "PropagationProblem", flow: CoupledFlowTables) -> None:
    """Refuse a problem whose surroundings the coupled flow cannot be executed in.

    The cumulative diagonal ``Phi`` multiplies the flow action, which is only the same
    equation when the two **commute**, so ``Phi`` must be a multiple of the identity in
    channels; a shot-noise draw kept apart is transported by the unitary phase of a
    diagonal flow, which a nonunitary coupled flow is not; the flux of a condensed
    boundary block is the square-root model's, whose positivity is not established, so
    no explicit metric may sit beside one on the mode-set route; and the block is
    **scalar**: the action contracts the channel axis alone, so the state carries one
    polarisation. The TE/TM wall block (``CoherentWall(impedance="te_tm")``) is scalar
    in this sense too: its families are channels. The **transform** may resolve two
    transverse components, since the synthesis is read by the responses alone.
    """
    _, n_pol, _ = problem.shape
    if n_pol != 1:
        msg = (
            "solprop executes a coupled flow only for a scalar state, and this "
            f"problem's state carries n_pol = {n_pol} polarisation components (the "
            "product route): the flow's action contracts the channel axis alone, so "
            "a scalar block beside a state with two polarisation components would be "
            "applied to each as though it described both. "
            "CoherentWall(impedance='te_tm') names the wall's TE and TM impedances, "
            "not the state's polarisation axis: its families are channels, one "
            "component each"
        )
        raise TranslationError(msg)
    linear = problem.linear
    constant = flow.section_methods == ("constant_schur",)
    if constant and (
        linear.phi_representation != "constant" or linear.phi_constant is None
    ):
        msg = (
            "solprop executes a constant-Schur coupled flow only beside the exact "
            f"constant representation of Phi, and this problem's is "
            f"{linear.phi_representation!r}: a varying carrier belongs to a "
            "'frozen_residual' flow"
        )
        raise TranslationError(msg)
    if not _mode_independent(linear):
        msg = (
            "solprop executes a coupled flow only beside a mode-independent Phi: "
            "the flow action and the cumulative diagonal are applied as a product, "
            "which is the same equation only when they commute, and a channel-"
            "dependent diagonal does not commute with a coupled block"
        )
        raise TranslationError(msg)
    if not constant and not problem.device_ready:
        msg = (
            "solprop executes a frozen-residual coupled flow only beside a Phi "
            "that is a weighted sum of fixed terms, and this problem's took the z "
            "table: the flow's own stage weights and the carrier's must come from "
            "one stage bundle, and the table has no such form "
            "(docs/dev/kernel-contract.md section 1)"
        )
        raise TranslationError(msg)
    # The input model's draw was added to the state when it was loaded, so the flow
    # carries it as part of the field (SEP 37 section 6); only the modified model keeps
    # a draw apart that a flow would have to transport.
    if problem.initial.noise_model == "modified":
        msg = (
            "solprop executes no 'modified' noise model beside a coupled flow: "
            "its one-photon-per-bin draw is advanced by the unitary phase of a "
            "diagonal flow, and its transport through a nonunitary coupled flow is "
            "not measured in this package; the 'input' model is admitted"
        )
        raise TranslationError(msg)
    if linear.modal_metric is not None and problem.transform.modal_tables is None:
        # The coherent wall's own flux model: a condensed boundary block has no bound
        # reciprocal basis. A general modal guide's metric is the Gram of its own
        # sampled, power-orthonormal basis, and the coupled block there a declared
        # unitary rotation of it, which leaves the Gram's spectrum where it was;
        # `transform.modal_tables` tells the two apart.
        msg = (
            "solprop executes no explicit metric beside a coupled flow on the "
            "mode-set route: the flux of a condensed boundary block is the "
            "square-root model's, whose positivity is not established. A general "
            "modal guide carries "
            "transform.modal_tables and its metric is its own sampled basis Gram, "
            "which this problem does not"
        )
        raise TranslationError(msg)


def _mode_independent(linear: LinearOperator) -> bool:
    """Return whether the cumulative ``Phi`` is a multiple of the identity in channels.

    That, and not constancy, is what the product ``exp(dPhi) exp(dz G_s)`` needs. Every
    representation is checked in its own terms -- the exact constant array, the
    expansion's fixed terms, or the table's coefficients -- because a varying guide
    takes whichever of them its declared parameters allow.
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


def _check_modal_route(problem: "PropagationProblem") -> None:
    """Refuse every general modal problem solprop does not execute.

    The route is one combination: a field grid (which the leaves' frequency axes
    already require), the tables' own execution maps, an
    **identity** metric -- the Poynting-normalised orthogonal basis the Gram gate
    certifies -- or the dense metric of a declared unitary change of basis of one, the
    constant unit power weight of a true-power amplitude, and the route's own
    responses. A legacy response beside them is refused, because the route's source
    replaces the node pipeline that would read it: it would be recorded and silently
    not propagated.
    """
    tables = problem.transform.modal_tables
    metric = problem.linear.modal_metric
    if tables is None:
        if metric is not None:
            msg = (
                "solprop executes linear.modal_metric as part of the general modal "
                "route, and this problem carries no transform.modal_tables for it to "
                "belong to"
            )
            raise TranslationError(msg)
        return
    if metric is None or metric.kind not in ("identity", "dense"):
        kind = "absent" if metric is None else f"of kind {metric.kind!r}"
        msg = (
            "solprop executes a general modal guide in a basis certified "
            "orthonormal in true power -- an identity linear.modal_metric, or the "
            "dense metric of a declared unitary change of basis of one -- and this "
            f"problem's is {kind}; a diagonal metric is a power-orthogonal basis "
            "with unequal channel norms, which nothing in this package builds and "
            "no backend has measured"
        )
        raise TranslationError(msg)
    defect = float(np.max(metric.error))
    if defect > DEFAULT_GRAM_ATOL:
        # An identity metric is executed as the identity and nothing downstream reads
        # the error row again, so an ungated certificate is an unbounded silent error.
        msg = (
            f"linear.modal_metric.error is {defect:.4g}, above the "
            f"{DEFAULT_GRAM_ATOL:.4g} solprop will call the identity: this basis "
            "is that far from orthonormal where it was sampled, and a continuous "
            "wave's da/dz would be wrong by about that much"
        )
        raise TranslationError(msg)
    _check_dense_metric(tables, metric)
    if tables.synthesis is None or tables.projection is None:
        msg = (
            "solprop executes a general modal guide through the synthesis and "
            "projection maps of transform.modal_tables, and this problem carries "
            "neither: a reference sampled over a path of positions has no execution "
            "map"
        )
        raise TranslationError(msg)
    if tables.synthesis.shape[0] != 1 or tables.synthesis.shape[1] != 1:
        _check_moving_maps(problem)
    weight = problem.linear.power_weight_constant
    if weight is None or not np.all(weight == 1.0):
        msg = (
            "solprop executes a general modal guide whose Poynting amplitude is "
            "already the true-power amplitude, so linear.power_weight is the "
            "constant one; this problem's is "
            f"{problem.linear.power_weight_representation}"
        )
        raise TranslationError(msg)
    _check_legacy_responses(problem)
    check_regional_responses(problem.responses.regional_responses)


def _check_dense_metric(
    tables: ModalTransformTables, metric: ModalMetricTables
) -> None:
    """Refuse a dense metric that is not a unitary change of basis, measuring it.

    The route propagates through the **diagonal** cumulative flow and measures the step
    controller's norm on the coordinate amplitudes, both of which are the basis's own
    only where that basis is orthonormal in true power. A declared unitary change of
    basis of a certified orthonormal reference stays there -- ``W' = I + U.H delta U``
    has the same spectral departure as ``delta`` -- and a nonunitary one does not.
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
        raise TranslationError(msg)
    departure = max(
        metric_departure(block[interval, degree])
        for interval in range(block.shape[0])
        for degree in range(block.shape[1])
    )
    if departure > MAX_DENSE_METRIC_DEPARTURE:
        msg = (
            f"linear.modal_metric is dense and {departure:.4g} from the identity, "
            f"above the {MAX_DENSE_METRIC_DEPARTURE:.4g} solprop will execute. "
            "The route propagates through the diagonal cumulative flow and "
            "measures the controller's norm on the coordinate amplitudes, so it "
            "admits a dense metric only as a unitary change of basis of a "
            "certified orthonormal reference, whose departure is the reference's "
            "own; a nonunitary transform leaves a genuinely nonorthogonal basis, "
            "whose flow is linear.coupled_flow and whose transport is unmeasured"
        )
        raise TranslationError(msg)


def _check_moving_maps(problem: "PropagationProblem") -> None:
    """Refuse a moving guide's maps beside anything but the flow of their connection.

    Moving maps with no connection are a guide whose modes change shape and exchange no
    power, which is the adiabatic approximation the route exists to replace.
    """
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
            "solprop executes a moving guide's maps beside the flow that carries "
            f"its connection, 'frozen_residual' on every section, and this problem "
            f"carries {described}: maps that move with no connection are the "
            "adiabatic approximation the moving route replaces"
        )
        raise TranslationError(msg)


def _check_legacy_responses(problem: "PropagationProblem") -> None:
    """Refuse a legacy response a general modal guide would carry and not propagate."""
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
        )
        if present
    ]
    if legacy:
        msg = (
            "solprop executes transform.modal_tables with its own "
            "responses.regional_responses and no legacy response, and this problem "
            f"also carries responses.{', responses.'.join(legacy)}: the general "
            "route's source replaces the node pipeline that would read them, so they "
            "would be recorded and not propagated"
        )
        raise TranslationError(msg)

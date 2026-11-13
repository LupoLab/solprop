# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled flows of a capillary's mode set: the coherent wall and the taper.

A coherent wall block (scalar or vector, on a constant or a varying guide) and the
taper coupling of a multimode capillary keep a shared scalar carrier in the cumulative
diagonal ``Phi`` and put the modal part of the flow in ``linear.coupled_flow``
(``docs/dev/kernel-contract.md`` section 3), with the generators, connections and
passivity checks they are built from.
"""

import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import numpy as np
from jaxtyping import Complex, Float

from solprop.constants import C
from solprop.exceptions import (
    GrazingIncidenceWarning,
    MaterialRangeWarning,
    TranslationError,
)
from solprop.grid import TimeGrid
from solprop.linear import Frame
from solprop.modes.capillary import (
    CoherentWall,
    MarcatiliMode,
    VectorWallOperator,
    WallImpedance,
    WallLimits,
    WallOperator,
    taper_connection,
    vector_wall_operator,
    wall_connection,
    wall_operator,
)
from solprop.modes.protocol import Mode
from solprop.translate.coupled import (
    constant_schur_flow,
    frozen_residual_flow,
    section_expansion,
)
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.linop import (
    LinearOperator,
    build_phi,
    delta_phi,
    with_power_weight,
)
from solprop.translate.validation import (
    _INPUT_NOISE_ADMITTED,
    DEFAULT_FLOW_CONDITION_LIMIT,
    TAPER_FLOW_CERTIFICATE,
    _Assembly,
)

__all__: list[str] = []

logger = logging.getLogger(__name__)


def _coupled_assembly(
    carried: Sequence[Mode],
    modes: Sequence[Mode],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
    *,
    index: float | None,
    reuse: ParameterExpansion | None = None,
    kinks: Sequence[float] = (),
) -> tuple[LinearOperator, LinearOperator]:
    """Return ``Phi`` of the carried channels, and it with the modes' power weight.

    The one table-to-operator site of every mode-set route: the run's span, axis,
    collar, loss policy and tolerances come from `build`. Beside a coupled flow the
    carried channels are copies of one channel, so that ``Phi`` is a multiple of the
    identity in the channel axis and commutes with the flow, while the power weight is
    always each mode's own.

    - ``index``: the constant index override of the operator and the weight, or
      ``None`` for each mode's ``Re n_eff``.
    - ``reuse``: a prequalified expansion of the same operator function, or ``None``.
    - ``kinks``: response positions in m that ``Phi`` must land on.
    - Returns: ``Phi`` alone, and the operator carrying the power weight.
    """
    carrier = build_phi(
        carried,
        grid,
        frame,
        z_max=build.z_max,
        n_pol=build.n_pol,
        absorber_rate=None if build.absorber is None else build.absorber.spectral_rate,
        loss=build.loss,
        margin=build.margin,
        tol=build.tol,
        reuse=reuse,
        index=index,
        kinks=kinks,
    )
    return carrier, with_power_weight(carrier, modes, grid, index)


def _wall_assembly(
    wall: CoherentWall,
    modes: Sequence[Mode],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> LinearOperator:
    """Assemble the linear operator of a capillary carrying a coherent wall block.

    The split is the one a constant section takes: the
    **shared scalar** part of the flow stays in the cumulative diagonal ``Phi``, where
    it is integrated analytically and commutes with the whole channel block, and the
    modal part -- the paraxial transverse term and the rank-one wall together -- is
    the coupled flow. Nothing may be split between them that does not commute, and a
    mode-dependent diagonal does not: ``exp(A) exp(B)`` is ``exp(A + B)`` only when
    ``A`` is a multiple of the identity, so ``Phi`` is built over ``N`` copies of the
    **first** channel and carries no mode dependence at all.

    What is subtracted from the block is therefore exactly what ``Phi`` applies per
    metre, and the co-moving frame cancels between the two: with ``k`` the in-core
    wavenumber and ``beta_11`` the first channel's own Marcatili propagation constant,
    the stored generator is ``conj(G + i(k - beta_11) I)``, the complex conjugate of
    the **whole** optics-convention equation (``docs/dev/conventions.md`` section 2).
    Its first diagonal entry is then ``conj(i(beta_wall - beta_11))``, which is the
    difference between the block's own HE_11 and this package's Marcatili one and is
    small by a derived bound, rather than either of them twice.
    The per-channel true-power weight is unaffected: it is a row of its own and is
    built from the real mode set, so a saved amplitude still carries each channel's
    own ``sqrt(Re n_eff)``.

    - ``modes``: the contiguous ``HE_11`` to ``HE_1N`` block in state order.
    - ``grid``: the run's grid; supplies the frequency axis and the active band.
    """
    block = tuple(modes)
    _check_wall_route(build)
    if wall.impedance == "te_tm":
        return _vector_wall_assembly(wall, block, grid, frame, build)
    if not cast("MarcatiliMode", block[0]).z_constant:
        return _varying_wall_assembly(wall, block, grid, frame, build)
    operator = wall_operator(
        cast("Sequence[MarcatiliMode]", block),
        grid.omega,
        z=0.0,
        active=grid.active_mask,
        scored=grid.inband_mask,
        validity_limit=wall.validity_limit,
    )
    carrier, linear = _coupled_assembly(
        (block[0],) * len(block), block, grid, frame, build, index=build.prefactor_index
    )
    # The carrier is built from N copies of one channel this function has already
    # established to be `z_constant` -- a varying one took `_varying_wall_assembly`
    # above -- and `build_phi` takes `constant_phi` for such a set whatever else it
    # is asked for, so this is an invariant of the constant route rather than a
    # condition on the run. It is asserted, not refused, because nothing can reach
    # the refusal; the varying route's own carrier is checked where it is built.
    assert carrier.phi_representation == "constant"
    assert carrier.phi_constant is not None
    generator = _wall_shifted_generator(
        operator.generator, cast("MarcatiliMode", block[0]), grid
    )
    _check_wall_total(generator, linear, len(block))
    flow = constant_schur_flow(
        grid.omega,
        np.asarray(carrier.phi_breakpoints, dtype=float)[[0, -1]],
        generator,
        dz_max=build.dz_max,
        condition_limit=wall.condition_limit,
        wall_vectors=operator.trace_vector,
        wall_weights=np.conj(operator.coupling),
        gauge_id="capillary HE_1m",
        certificate_id="constant-schur",
    )
    logger.debug(
        "coherent wall: %d channels, worst validity ratio %.3e (%s), largest "
        "conditioning %.3e over dz_max %.3g m, representation error %.3e",
        len(block),
        operator.limits.worst,
        operator.limits.named()[0],
        float(np.max(flow.inverse_condition_bound)),
        build.dz_max,
        float(np.max(flow.generator_error)),
    )
    return replace(linear, coupled_flow=flow)


def _vector_wall_assembly(
    wall: CoherentWall,
    block: tuple[Mode, ...],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> LinearOperator:
    """Assemble the linear operator of a capillary carrying a vector wall block.

    The split is the scalar wall's, unchanged: the shared scalar carrier stays in the
    cumulative diagonal ``Phi``, built over ``N`` copies of the **first** channel so
    that it is a multiple of the identity and commutes with the block, and the whole
    modal part -- the paraxial transverse term and the **rank-two** wall together --
    is the coupled flow (:func:`solprop.modes.capillary.vector_wall_operator`). What
    differs is the block itself and two admissions.

    **No wall pair is stored.** The leaf's optional factored pair is one vector and
    one weight, which reconstructs a rank-**one** off-diagonal; the vector block's
    wall term is ``V W V^T`` with two columns. Storing half of it would be worse than
    storing none, so the dense generator stands alone and the constructor is given no
    pair to check.

    **A guide that varies along z** takes the frozen-residual route
    (:func:`_varying_wall_assembly`) with this block in place of the scalar one and
    the vector connection, which is block diagonal -- each family's own ``C_mn`` of
    its Bessel order, and no term between the families -- as derived and as measured
    (:func:`solprop.modes.capillary.wall_connection`).
    """
    first = cast("MarcatiliMode", block[0])
    if not first.z_constant:
        return _varying_wall_assembly(wall, block, grid, frame, build)
    operator = vector_wall_operator(
        block,
        grid.omega,
        z=0.0,
        active=grid.active_mask,
        scored=grid.inband_mask,
        validity_limit=wall.validity_limit,
    )
    carrier, linear = _coupled_assembly(
        (first,) * len(block), block, grid, frame, build, index=build.prefactor_index
    )
    # As on the constant scalar route: the carrier is N copies of a channel already
    # established to be `z_constant`, and `build_phi` takes `constant_phi` for such a
    # set whatever else it is asked for, so this is an invariant and not a condition.
    assert carrier.phi_representation == "constant"
    generator = _wall_shifted_generator(operator.generator, first, grid)
    _check_wall_total(generator, linear, len(block))
    flow = constant_schur_flow(
        grid.omega,
        np.asarray(carrier.phi_breakpoints, dtype=float)[[0, -1]],
        generator,
        dz_max=build.dz_max,
        condition_limit=wall.condition_limit,
        gauge_id=f"capillary azimuthal order nu={operator.sector}",
        certificate_id="constant-schur",
    )
    logger.debug(
        "coherent wall, vector: nu = %d, %s, %d channels, worst validity ratio %.3e "
        "(%s), anisotropy %.3e of the mean impedance, largest conditioning %.3e over "
        "dz_max %.3g m, representation error %.3e",
        operator.sector,
        "+".join(family.label for family in operator.families),
        len(block),
        operator.limits.worst,
        operator.limits.named()[0],
        float(
            np.max(
                abs(operator.impedance[0] - operator.impedance[1])[operator.active]
                / abs(operator.impedance[0] + operator.impedance[1])[operator.active]
            )
        ),
        float(np.max(flow.inverse_condition_bound)),
        build.dz_max,
        float(np.max(flow.generator_error)),
    )
    return replace(linear, coupled_flow=flow)


def _varying_wall_assembly(
    wall: CoherentWall,
    block: tuple[Mode, ...],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> LinearOperator:
    """Assemble the coherent wall of a capillary whose geometry or fill varies along z.

    The **complete** generator of each section -- the paraxial transverse term, the
    rank-one wall and the carrier shift together -- is expanded in Chebyshev
    coefficients along z; the block frozen inside the exponential flow is that
    expansion at the section's midpoint, and

    ``R_s(z) = D(z) - D(z_s) + H(z) - H(z_s) - C(z)``

    is the explicit residual the interaction-picture source carries, with ``C`` the
    HE1m moving-basis connection
    (:func:`solprop.modes.capillary.wall_connection`). With ``R`` evaluated exactly,
    freezing is a rearrangement and not an approximation. The **vector** block of
    ``impedance="te_tm"`` takes the same route unchanged but for the block
    (:func:`solprop.modes.capillary.vector_wall_operator`) and its connection, which
    is each radial family's own and nothing between them.

    Two things differ from the constant section of a guide that does not vary. The
    cumulative ``Phi`` is
    no longer the exact constant representation but the **parameter expansion** of the
    lowest channel, which is still a multiple of the identity in the channel axis and
    therefore still commutes with the block -- that, and not constancy, is what the
    product of the two exponentials needs; the rest of each channel's own
    normalisation is on the generator's diagonal (:func:`_channel_normalisation_at`).
    And the section edges are the operator
    table's own breakpoints, which already land on every declared kink of the radius
    and the density, subdivided ``wall.sections`` times: a step that crosses one is
    still the same equation, because the residual is defined against whichever block
    the step's anchor names, so the edges need not be landed on exactly.

    - ``block``: the contiguous ``HE_11`` to ``HE_1N`` set in state order, or one
      azimuthal order of the capillary for ``wall.impedance == "te_tm"``.
    """
    first = cast("MarcatiliMode", block[0])
    carrier, linear = _coupled_assembly(
        (first,) * len(block), block, grid, frame, build, index=build.prefactor_index
    )
    edges = _section_edges(np.asarray(carrier.phi_breakpoints, dtype=float), wall)
    sampled: list[tuple[float, WallLimits]] = []
    coefficients, tail = section_expansion(
        lambda positions: np.stack(
            [
                _wall_generator_at(block, grid, float(z), sampled, wall.impedance)
                for z in positions
            ]
        ),
        edges,
        name="the coherent wall's generator",
    )
    link = None
    if not first.radius_is_constant:
        link, link_tail = section_expansion(
            lambda positions: np.stack(
                [
                    _wall_connection_at(block, grid, float(z), wall.impedance)
                    for z in positions
                ]
            ),
            edges,
            name="the coherent wall's taper connection",
        )
        tail = max(tail, link_tail)
        # One rectangular degree axis serves both: a Chebyshev series padded with
        # zeros is the same function, so whichever resolved at a lower degree is
        # lengthened rather than either being rebuilt.
        degrees = max(coefficients.shape[1], link.shape[1])
        coefficients = _padded_degrees(coefficients, degrees)
        link = _padded_degrees(link, degrees)
    # The admission is over every position the expansion was built from, and the one
    # call that carries the caller's limit is at the worst of them, so a varying guide
    # is refused or warned about once and names the geometry that did it.
    position = sampled[int(np.argmax([limits.worst for _, limits in sampled]))][0]
    operator = _wall_block_at(
        block, grid, position, wall.validity_limit, wall.impedance
    )
    gauge_id = (
        "capillary HE_1m"
        if isinstance(operator, WallOperator)
        else f"capillary azimuthal order nu={operator.sector}"
    )
    _check_wall_total_varying(coefficients, linear, edges, block, grid, tail)
    flow = frozen_residual_flow(
        grid.omega,
        edges,
        coefficients,
        link,
        dz_max=build.dz_max,
        condition_limit=wall.condition_limit,
        representation_error=tail,
        gauge_id=gauge_id,
        certificate_id="frozen-residual",
    )
    logger.debug(
        "coherent wall, varying (%s): %d channels over %d section(s), worst validity "
        "ratio %.3e (%s) at z = %.6g m, residual bound %.3e 1/m, largest conditioning "
        "%.3e over dz_max %.3g m, representation error %.3e",
        gauge_id,
        len(block),
        len(flow.section_methods),
        operator.limits.worst,
        operator.limits.named()[0],
        position,
        float(np.max(flow.residual_norm_bound)),
        float(np.max(flow.inverse_condition_bound)),
        build.dz_max,
        float(np.max(flow.generator_error)),
    )
    return replace(linear, coupled_flow=flow)


def _taper_assembly(
    block: tuple[MarcatiliMode, ...],
    grid: TimeGrid,
    frame: Frame,
    build: _Assembly,
) -> LinearOperator:
    """Assemble a multimode capillary taper's coupled flow without a wall.

    The varying coherent wall's route (:func:`_varying_wall_assembly`) with the wall
    term zero, which is the whole of the derivation: in a basis attached to the local
    radius the equation of the true-power amplitude is ``b' = (i B(z) - C(z)) b``,
    with ``B`` the channels' own propagation constants and ``C`` the HE1m connection
    (:func:`solprop.modes.capillary.taper_connection`). Without the wall's rank-one
    block the diagonal is the **modes' own** ``beta_n(z)`` -- the Marcatili index of
    whatever model and loss the channels declare, lossless or not -- rather than the
    wall block's paraxial condensation of it, so that with ``C`` dropped the flow
    propagates exactly what the diagonal route propagates. The cumulative ``Phi``
    carries the lowest channel for every channel, as beside a wall; the flow's
    generator is each channel's rate beyond it, ``conj(i(beta_n - beta_1))``, with
    each channel's normalisation completed (:func:`_channel_normalisation_at`),
    frozen per section of the carrier's own table with the variation and the
    connection in the explicit residual (:func:`frozen_residual_flow`).

    - ``block``: the contiguous ``HE_11`` to ``HE_1N`` set on a varying radius, from
      :func:`solprop.modes.capillary.taper_coupling_block`.
    """
    _check_taper_route(build, grid)
    # The block's own door, before the generator differentiates the profiles: it
    # names an unslope-able radius or disagreeing channels, and the opt-out.
    taper_connection(block)
    first = block[0]
    carrier, linear = _coupled_assembly(
        (first,) * len(block),
        block,
        grid,
        frame,
        build,
        index=build.prefactor_index,
        kinks=[z for z, _ in build.kinks],
    )
    edges = np.asarray(carrier.phi_breakpoints, dtype=float)
    coefficients, tail = section_expansion(
        lambda positions: np.stack(
            [_taper_generator_at(block, grid, float(z)) for z in positions]
        ),
        edges,
        name="the taper coupling's generator",
    )
    link, link_tail = section_expansion(
        lambda positions: np.stack(
            [
                _true_power_connection(
                    taper_connection(block, float(z)), block, grid, float(z)
                )
                for z in positions
            ]
        ),
        edges,
        name="the taper connection",
    )
    tail = max(tail, link_tail)
    degrees = max(coefficients.shape[1], link.shape[1])
    coefficients = _padded_degrees(coefficients, degrees)
    link = _padded_degrees(link, degrees)
    _check_wall_total_varying(coefficients, linear, edges, block, grid, tail)
    try:
        flow = frozen_residual_flow(
            grid.omega,
            edges,
            coefficients,
            link,
            dz_max=build.dz_max,
            condition_limit=DEFAULT_FLOW_CONDITION_LIMIT,
            representation_error=tail,
            gauge_id="capillary HE_1m",
            certificate_id=TAPER_FLOW_CERTIFICATE,
        )
    except TranslationError as error:
        # The flow's own refusals name its section count and its limit, which are a
        # wall's declarations; the taper coupling has neither, and has the opt-out.
        msg = (
            f"{error}. This is the taper coupling of a multimode capillary, "
            "which declares no sections and no limit: cap "
            "dz_max, or pass taper_coupling=False for the adiabatic route"
        )
        raise TranslationError(msg) from error
    logger.debug(
        "taper coupling: %d channels over %d section(s), residual bound %.3e 1/m, "
        "largest conditioning %.3e over dz_max %.3g m, representation error %.3e",
        len(block),
        len(flow.section_methods),
        float(np.max(flow.residual_norm_bound)),
        float(np.max(flow.inverse_condition_bound)),
        build.dz_max,
        float(np.max(flow.generator_error)),
    )
    return replace(linear, coupled_flow=flow)


def _check_taper_route(build: _Assembly, grid: TimeGrid) -> None:
    """Refuse, by name, a run the taper coupling cannot be carried in.

    - ``grid``: the run's grid, which must be a field grid.
    """
    if grid.kind != "field":
        msg = (
            "the taper coupling of a multimode capillary is a coupled flow tabulated "
            "on an increasing frequency axis, which a field grid has and an "
            f"{grid.kind} grid, in FFT order about its carrier, does not; no coupled "
            "flow is built on one in this package. Pass envelope=False, or "
            "taper_coupling=False for the adiabatic route"
        )
        raise TranslationError(msg)
    if build.n_pol != 1:
        msg = (
            f"the taper coupling of a multimode capillary carries one field "
            f"component and this run carries n_pol = {build.n_pol}: the coupled "
            "flow's action contracts the channel axis alone. Pass "
            "taper_coupling=False for the adiabatic route"
        )
        raise TranslationError(msg)
    if build.noise_model == "modified":
        msg = (
            "the taper coupling of a multimode capillary refuses the 'modified' "
            "noise model: its draw is transported by the unitary phase of a "
            "diagonal flow, and its transport through a coupled flow is not "
            "measured in this package. Pass taper_coupling=False for the adiabatic "
            "route, whose diagonal flow carries the draw. " + _INPUT_NOISE_ADMITTED
        )
        raise TranslationError(msg)


def _taper_generator_at(
    block: tuple[MarcatiliMode, ...], grid: TimeGrid, z: float
) -> Complex[np.ndarray, "n_modes_t_out n_modes_t_in n_omega_t"]:
    """Return the taper coupling's internal-convention generator at one position.

    Diagonal: each channel's own propagation constant beyond the lowest channel's,
    which the cumulative ``Phi`` carries for every channel, conjugated as the whole
    optics-convention equation is (``docs/dev/conventions.md`` section 2), so
    ``conj(i(beta_n - beta_1))`` -- exactly zero for the lowest channel -- plus the
    rest of each channel's normalisation rate (:func:`_channel_normalisation_at`).
    The difference is formed from ``n_eff - 1`` without the round trip through unity.

    - ``block``: the contiguous ``HE_11`` to ``HE_1N`` set.
    - ``z``: position in m.
    - Returns: the generator in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``,
      exactly zero off the grid's active band.
    """
    band = grid.omega[grid.active_mask]
    reference = np.asarray(block[0].neff_minus_one(band, z))
    count = len(block)
    generator = np.zeros((count, count, grid.n_omega), dtype=np.complex128)
    for index, mode in enumerate(block[1:], start=1):
        rate = (np.asarray(mode.neff_minus_one(band, z)) - reference) * band / C
        generator[index, index, grid.active_mask] = np.conj(1j * rate)
    generator[np.arange(count), np.arange(count)] += _channel_normalisation_at(
        block, grid, z
    )
    return generator


def _section_edges(
    breakpoints: Float[np.ndarray, " n_breakpoints"], wall: CoherentWall
) -> Float[np.ndarray, " n_edges"]:
    """Return the flow's section edges: the table's own, subdivided.

    The **placement** is the operator table's, which already lands on every declared
    kink of the radius and the density and never spans one; `CoherentWall.sections` is
    the refinement inside each of those, in equal parts.

    - ``breakpoints``: the carrier's interval edges in m, strictly increasing.
    - Returns: the edges in m, strictly increasing, with the same two ends.
    """
    if wall.sections == 1:
        return np.array(breakpoints, dtype=float)
    fractions = np.arange(wall.sections) / wall.sections
    inner = (
        breakpoints[:-1, None] + fractions[None, :] * np.diff(breakpoints)[:, None]
    ).reshape(-1)
    return np.concatenate([inner, breakpoints[-1:]])


def _padded_degrees(
    coefficients: Complex[np.ndarray, "n_intervals_p n_deg_p *rest_p"], degrees: int
) -> np.ndarray:
    """Return a coefficient block lengthened along its degree axis with exact zeros.

    - ``degrees``: the degree axis to reach, at least the block's own.
    - Returns: the block, with the same values and a degree axis of `degrees`.
    """
    pad = [(0, 0)] * coefficients.ndim
    pad[1] = (0, degrees - coefficients.shape[1])
    return np.pad(coefficients, pad)


def _wall_block_at(
    block: tuple[Mode, ...],
    grid: TimeGrid,
    z: float,
    validity_limit: float,
    impedance: WallImpedance,
) -> WallOperator | VectorWallOperator:
    """Return the wall block the declared impedance names, at one position.

    - ``z``: position in m.
    """
    if impedance == "mean":
        return wall_operator(
            cast("Sequence[MarcatiliMode]", block),
            grid.omega,
            z=z,
            active=grid.active_mask,
            scored=grid.inband_mask,
            validity_limit=validity_limit,
        )
    return vector_wall_operator(
        block,
        grid.omega,
        z=z,
        active=grid.active_mask,
        scored=grid.inband_mask,
        validity_limit=validity_limit,
    )


def _wall_generator_at(
    block: tuple[Mode, ...],
    grid: TimeGrid,
    z: float,
    sampled: list[tuple[float, WallLimits]],
    impedance: WallImpedance = "mean",
) -> Complex[np.ndarray, "n_modes_s_out n_modes_s_in n_omega_s"]:
    """Return the complete internal-convention wall generator at one position.

    The same split as the constant section, evaluated at `z`: the stored generator is
    ``conj(G + i(k - beta_11) I)``, so that what the cumulative ``Phi`` applies per
    metre is exactly what is subtracted here and the co-moving frame cancels between
    the two. Both the block and the reference channel are read **at this position**,
    which is the whole difference from the constant section.

    - ``block``: the contiguous ``HE_11`` to ``HE_1N`` set.
    - ``z``: position in m.
    - ``sampled``: collects the limits of every position sampled, for the admission
      that follows.
    - ``impedance``: which block to sample, ``"mean"`` or ``"te_tm"``.
    - Returns: the generator in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``,
      exactly zero off the active band.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GrazingIncidenceWarning)
        warnings.simplefilter("ignore", MaterialRangeWarning)
        operator = _wall_block_at(block, grid, z, math.inf, impedance)
    sampled.append((z, operator.limits))
    generator = _wall_shifted_generator(
        operator.generator, cast("MarcatiliMode", block[0]), grid, z
    )
    count = len(block)
    generator[np.arange(count), np.arange(count)] += _channel_normalisation_at(
        block, grid, z
    )
    return generator


def _channel_normalisation_at(
    block: tuple[Mode, ...], grid: TimeGrid, z: float
) -> Float[np.ndarray, "n_modes_q n_omega_q"]:
    """Return each channel's normalisation rate beyond the reference channel's.

    The cumulative ``Phi`` beside a coupled block is built from the lowest channel
    alone, so the term that renormalises a vacuum amplitude along a varying index,
    ``-0.5 ln[w(z)/w(0)]`` with ``w = Re n_eff``, is channel 1's for every
    channel. Each channel's own is completed here, as the rest of its rate:
    ``-0.5 d ln(w_n/w_1)/dz = -0.5 (w_n'/w_n - w_1'/w_1)``, real, so the
    optics/internal conjugation leaves it alone, zero for channel 1 by construction
    and zero where the guide does not vary. It is part of the generator, so the
    frozen block carries it at the section midpoint and the explicit residual carries
    its variation. The slope is the analytic
    :meth:`solprop.modes.capillary.MarcatiliMode.dneff_dz`, never a difference.

    - ``block``: the contiguous ``HE_11`` to ``HE_1N`` set.
    - ``z``: position in m.
    - Returns: the rate in 1/m, shape ``(n_modes, n_omega)``, exactly zero off the
      grid's active band.
    """
    band = grid.omega[grid.active_mask]
    logarithmic = np.stack(
        [
            np.real(cast("MarcatiliMode", mode).dneff_dz(band, z))
            / np.real(np.asarray(mode.neff(band, z)))
            for mode in block
        ]
    )
    rate = np.zeros((len(block), grid.n_omega))
    rate[:, grid.active_mask] = -0.5 * (logarithmic - logarithmic[:1])
    return rate


def _wall_shifted_generator(
    generator: Complex[np.ndarray, "n_modes_g_out n_modes_g_in n_omega_g"],
    first: MarcatiliMode,
    grid: TimeGrid,
    z: float = 0.0,
) -> Complex[np.ndarray, "n_modes_g2_out n_modes_g2_in n_omega_g2"]:
    """Return a wall block's generator in the internal convention, with its carrier.

    ``Gamma_internal = conj(i(k - beta_0 - beta_1 dw) I + G)`` and the carrier ``Phi``
    applies ``conj(i(beta_11 - beta_0 - beta_1 dw))``; the co-moving frame cancels
    between them and what is left on the diagonal is ``conj(i(k - beta_11))``, the
    difference between the block's own in-core carrier ``k = n_core omega/c`` and the
    lowest channel's Marcatili propagation constant. The conjugation is of the
    **whole** equation (``docs/dev/conventions.md`` section 2): it is not ``-G``, and
    a passive channel propagated with that would grow.

    ``k - beta_11`` is formed as ``((n_core - 1) - (n_eff - 1)) omega/c``
    (:meth:`~solprop.modes.capillary.MarcatiliMode.core_index_minus_one`,
    :meth:`~solprop.modes.capillary.MarcatiliMode.neff_minus_one`), as the taper
    coupling forms its own diagonal, and never as the difference of the two
    wavenumbers: those agree to five or six digits, so their difference would carry
    the absolute rounding of ``k`` (about 1e-9 1/m) instead of that of the
    refractivities.

    - ``generator``: the optics-convention block in 1/m, shape ``(n_modes, n_modes,
      n_omega)``.
    - ``z``: position in m both are read at.
    - Returns: the internal-convention generator in 1/m, same shape.
    """
    band = grid.omega[grid.active_mask]
    refractivity = np.asarray(first.core_index_minus_one(band, z)) - np.asarray(
        first.neff_minus_one(band, z)
    )
    shift = np.zeros(grid.n_omega, dtype=np.complex128)
    shift[grid.active_mask] = refractivity * band / C
    shifted = np.conj(generator)
    count = shifted.shape[0]
    shifted[np.arange(count), np.arange(count)] += np.conj(1j * shift)
    return shifted


def _wall_connection_at(
    block: tuple[Mode, ...], grid: TimeGrid, z: float, impedance: WallImpedance = "mean"
) -> Complex[np.ndarray, "n_modes_c_out n_modes_c_in n_omega_c"]:
    """Return the wall block's connection at one position, on the run's frequency axis.

    :func:`solprop.modes.capillary.wall_connection` is the connection of the
    field-normalised Bessel basis, antisymmetric in the **true-power** amplitude
    ``b = W^(1/2) a`` with ``W = diag(Re n_eff)`` -- which is where coupled local-mode
    theory puts it, and what makes a lossless taper conserve power exactly. The
    kernel's channel amplitude is ``a``, so the stored term is
    ``W^(-1/2) C W^(1/2)``, entry by entry ``sqrt(w_n / w_m) C_mn``: real, so the
    optics/internal conjugation of ``docs/dev/conventions.md`` section 2 leaves it
    alone, and frequency dependent through the weights alone. It is exactly zero
    off the active band, because the
    residual it joins is, and a bin the wall formula was never evaluated at must carry
    no term at all. The vector block's connection (``impedance="te_tm"``) is carried
    the same way: each channel's weight is its own, whichever family it is in, and
    the terms between the families are zero before the weights and after them.

    - ``block``: the contiguous ``HE_11`` to ``HE_1N`` set, or one azimuthal order.
    - ``z``: position in m.
    - ``impedance``: which block the connection is of, ``"mean"`` or ``"te_tm"``.
    - Returns: ``C(z)`` in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``.
    """
    return _true_power_connection(
        wall_connection(block, z, impedance=impedance), block, grid, z
    )


def _true_power_connection(
    link: Float[np.ndarray, "n_modes_l_out n_modes_l_in"],
    block: tuple[Mode, ...],
    grid: TimeGrid,
    z: float,
) -> Complex[np.ndarray, "n_modes_w_out n_modes_w_in n_omega_w"]:
    """Return a connection carried to the kernel's amplitude on the run's band.

    ``W^(-1/2) C W^(1/2)`` with ``W = diag(Re n_eff)`` at `z`, entry by entry
    ``sqrt(w_n / w_m) C_mn``, exactly zero off the active band
    (:func:`_wall_connection_at` gives the reason).

    - ``link``: ``C`` in 1/m, real, shape ``(n_modes, n_modes)``.
    - ``z``: position in m.
    - Returns: the term in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``.
    """
    band = grid.omega[grid.active_mask]
    weight = np.stack([np.real(np.asarray(mode.neff(band, z))) for mode in block])
    root = np.sqrt(weight)
    scattered = np.zeros((*link.shape, grid.n_omega), dtype=np.complex128)
    scattered[:, :, grid.active_mask] = (
        link[:, :, None] * root[None, :, :] / root[:, None, :]
    )
    return scattered


def _check_wall_total_varying(  # noqa: PLR0917 - called positionally
    coefficients: Complex[
        np.ndarray, "n_intervals_v n_deg_v n_modes_v_out n_modes_v_in n_omega_v"
    ],
    linear: LinearOperator,
    edges: Float[np.ndarray, " n_edges_v"],
    block: tuple[Mode, ...],
    grid: TimeGrid,
    tail: float = 0.0,
) -> None:
    """Refuse a varying coupled route whose total internal generator gains on average.

    The varying counterpart of :func:`_check_wall_total`, and the same door: a
    conjugation applied to the block and not to the carrier turns the diagonal into
    gain. What is compared is each section's **mean** rate, because ``Phi`` has no
    pointwise rate this layer can read and its exact increment over the section does:
    the mean of a Chebyshev series on its own interval is
    ``c_0 - c_2/3 - c_4/15 - ...``, and ``delta_phi`` over the section divided by its
    width is the carrier's. A sign slipped into this arithmetic is a *global*
    property of the conjugation, which is what this catches; the block's own pointwise
    passivity is checked at every sampled position by
    :func:`solprop.modes.capillary.wall_operator`.

    - ``coefficients``: the stored generator's Chebyshev coefficients in 1/m.
    - ``edges``: the section edges in m.
    - ``block``: the mode set. ``Phi`` was built from its lowest channel and the flow's
      diagonal completes every other channel's normalisation, so the allowance is each
      channel's own.
    - ``tail``: the expansion's relative Chebyshev tail, dimensionless. The
      interpolant's section mean departs from the function's by about the size of its
      last coefficients, so twice the tail times the block's scale joins the rounding
      budget: the normalisation term is compared through the interpolant on one side and
      exactly on the other.
    """
    count = len(block)
    degrees = np.arange(coefficients.shape[1])
    # The mean of T_k on its own interval is zero for odd k and 1/(1 - k^2) for even,
    # which is one at k = 0 and never divides by zero, k = 1 being odd.
    means = np.zeros(degrees.size)
    even = degrees % 2 == 0
    means[even] = 1.0 / (1.0 - degrees[even] ** 2.0)
    scale = float(np.max(abs(coefficients)))
    budget = (64 * count * float(np.finfo(float).eps) + 2.0 * tail) * scale
    for index in range(edges.size - 1):
        lo, hi = float(edges[index]), float(edges[index + 1])
        carrier = np.real(delta_phi(linear, lo, hi))[0, :, 0, :] / (hi - lo)
        frozen = np.tensordot(means, coefficients[index], axes=1)
        allowance = budget + _normalisation_rate(block, grid, lo, hi)
        total = (
            carrier + np.real(frozen[np.arange(count), np.arange(count)]) - allowance
        )
        largest = float(np.max(total))
        if largest > 0.0:
            channel, bin_index = np.unravel_index(int(np.argmax(total)), total.shape)
            msg = (
                f"the coupled flow's total internal generator grows on "
                f"section {index} of [{lo:.6g}, {hi:.6g}] m: channel "
                f"{int(channel)} gains {largest:.4g} 1/m on average at bin "
                f"{int(bin_index)} beyond its own normalisation term and the "
                f"{budget:.4g} 1/m rounding of a block "
                f"whose own scale is {scale:.4g} 1/m. Crossing the optics/internal "
                "boundary conjugates the whole equation, carrier and block alike "
                "(docs/dev/conventions.md section 2)"
            )
            raise TranslationError(msg)


def _normalisation_rate(
    block: tuple[Mode, ...], grid: TimeGrid, lo: float, hi: float
) -> Float[np.ndarray, "n_modes_r n_omega_r"]:
    """Return the mean rate each channel's normalisation term adds over one section.

    ``Phi`` carries ``-0.5 ln[w(z)/w(0)]`` beside the integral of ``D``, which
    renormalises a vacuum amplitude along a varying effective index
    (``docs/dev/kernel-contract.md`` section 3), and on the wall route it is the lowest
    channel's; the flow's diagonal completes every other channel's to its own
    (:func:`_channel_normalisation_at`). The term is **not sign definite**: along a
    falling pressure the index falls and it is positive, so the total internal
    generator's real part is positive by that much with nothing wrong. The passivity
    guard allows exactly it, per channel, and nothing else -- a conjugation slipped
    into the block turns the diagonal into gain of the order of the whole wall loss,
    three orders above this -- and the allowance is each channel's term's own exact
    mean over the section, ``|ln(w_n(hi)/w_n(lo))|/(2 (hi - lo))``.

    - ``grid``: the run's grid; the allowance is zero off its active band, where the
      block is.
    - ``lo``, ``hi``: the section's ends in m.
    - Returns: the non-negative allowance in 1/m, shape ``(n_modes, n_omega)``.
    """
    band = grid.omega[grid.active_mask]
    rate = np.zeros((len(block), grid.n_omega))
    for channel, mode in enumerate(block):
        wide = np.real(np.asarray(mode.neff(band, lo)))
        narrow = np.real(np.asarray(mode.neff(band, hi)))
        ratio = np.divide(narrow, wide, out=np.ones_like(narrow), where=wide > 0.0)
        rate[channel, grid.active_mask] = abs(
            0.5 * np.log(np.where(ratio > 0.0, ratio, 1.0))
        ) / (hi - lo)
    return rate


def _check_wall_route(build: _Assembly) -> None:
    """Refuse, by name, every run a coherent wall block is not derived for.

    The two the *assembly* owns. The route the run takes is refused earlier still
    (:func:`_refuse_a_wall_off_the_capillary_route`, the first thing
    :func:`assemble` does), the mode set by
    :func:`solprop.modes.capillary.wall_operator`, and an empty active band by that
    same function, which names the mask it was given and is the layer that evaluates
    over it.
    """
    if build.n_pol != 1:
        msg = (
            f"a coherent wall block carries one field component and this run carries "
            f"n_pol = {build.n_pol}: the flow's action contracts the channel axis "
            "alone, so a block beside two components would be applied to each as "
            "though it described both. CoherentWall(impedance='te_tm') does not "
            "admit a second component either: it names the wall's own TE and TM "
            "impedances, not the state's polarisation axis, and its mode families "
            "are channels of the mode set, one component each"
        )
        raise TranslationError(msg)
    if build.noise_model == "modified":
        msg = (
            "a coherent wall block refuses the 'modified' noise model: its draw is "
            "transported by the unitary phase of a diagonal flow, and the transport "
            "of a draw through a nonunitary coupled flow is not measured in this "
            "package. " + _INPUT_NOISE_ADMITTED
        )
        raise TranslationError(msg)


def _check_wall_total(
    generator: Complex[np.ndarray, "n_modes_t_out n_modes_t_in n_omega_t"],
    linear: LinearOperator,
    count: int,
) -> None:
    """Refuse a wall route whose total internal generator shows gain.

    The last of the three doors the sign of this operator passes through. The wall
    block itself is checked for passivity in its own coordinates
    (:func:`solprop.modes.capillary.wall_operator`); here the **carrier** is added
    back, so a conjugation applied to one of the two terms and not the other -- the
    easy mistake of writing ``-i B`` where ``-i conj(B)`` was meant -- turns the
    diagonal into gain and is caught.

    - ``generator``: the stored internal-convention generator in 1/m.
    """
    assert linear.phi_constant is not None
    total = np.real(linear.phi_constant[0, :, 0, :]) + np.real(
        generator[np.arange(count), np.arange(count)]
    )
    scale = float(np.max(abs(generator))) + float(np.max(abs(linear.phi_constant)))
    budget = 64 * count * float(np.finfo(float).eps) * scale
    largest = float(np.max(total))
    if largest > budget:
        channel, bin_index = np.unravel_index(int(np.argmax(total)), total.shape)
        msg = (
            f"the coherent wall route's total internal generator grows: channel "
            f"{int(channel)} gains {largest:.4g} 1/m at bin {int(bin_index)}, above "
            f"the {budget:.4g} 1/m rounding of a problem whose own scale is "
            f"{scale:.4g} 1/m. Crossing the optics/internal boundary conjugates the "
            "whole equation, carrier and block alike (docs/dev/conventions.md "
            "section 2)"
        )
        raise TranslationError(msg)

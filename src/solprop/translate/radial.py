# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Translate radial Hankel geometry and bulk propagation into the spatial seam."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from solprop.boundaries import TEMPORAL_ALPHA_LENGTH, Absorber
from solprop.exceptions import TranslationError
from solprop.grid import RadialGrid, TimeGrid
from solprop.spatial import SpatialDispersion
from solprop.translate.chebyshev import (
    DEFAULT_TOL,
    ENVELOPE_FLOOR,
    PiecewiseChebyshev,
    build,
)
from solprop.translate.expansion import (
    MAX_PARAMETERS,
    ParameterExpansion,
    build_expansion,
    reweight_expansion,
)
from solprop.translate.linop import (
    DEFAULT_MARGIN,
    LinearOperator,
    constant_phi,
    tabulate,
)
from solprop.translate.modal import DEFAULT_NODE_BUDGET, Transform, _node_chunk
from solprop.translate.spatial import (
    RateBounds,
    SeparableOperator,
    SpatialGeometry,
    separable_values,
)

__all__ = ["radial_operators", "radial_rate_bounds", "radial_transform"]

logger = logging.getLogger(__name__)


def radial_transform(  # noqa: PLR0913 - geometry and the shared translation controls
    space: RadialGrid,
    time: TimeGrid,
    *,
    n_pol: int,
    z_max: float,
    absorber: Absorber | None = None,
    node_budget: int = DEFAULT_NODE_BUDGET,
    margin: float = DEFAULT_MARGIN,
) -> Transform:
    """Build the scalar inverse/forward QDHT applied to each state polarisation.

    Parameters
    ----------
    space : solprop.grid.RadialGrid
        Validated radial axes, weights and collar.
    time : solprop.grid.TimeGrid
        Temporal grid supplying the pad scaling and chunk-buffer size.
    n_pol : int
        Number of state polarisation components, 1 or 2.
    z_max : float
        Propagation length in m, positive.
    absorber : solprop.boundaries.Absorber or None, optional
        Supplies the spatial collar's reference length. None disables spatial loss.
    node_budget : int, optional
        Shared node-buffer budget in bytes, positive.
    margin : float, optional
        Relative extension of the scalar diagnostic table beyond z_max.

    Returns
    -------
    solprop.translate.modal.Transform
        Hankel matrices, geometry and bounded node chunk, ready for all backends.
    """
    space.validate()
    hankel = space.hankel
    diagonal = hankel.j1[:, None] / hankel.j1[None, :]
    synthesis = hankel.inverse_scale * diagonal * hankel.matrix * time.pad_scale
    projection = hankel.forward_scale * diagonal * hankel.matrix
    rate = (
        np.zeros(space.n_r)
        if absorber is None
        else (TEMPORAL_ALPHA_LENGTH * (1 - space.window) / absorber.in_stage_length)
    )
    geometry = SpatialGeometry(
        kind="radial",
        shape=(space.n_r,),
        axes=(space.r,),
        spectral_axes=(space.k_r,),
        spatial_weights=space.radial_weights,
        spectral_weights=space.spectral_weights,
        coordinates=np.column_stack((space.r, np.zeros(space.n_r))),
        window=space.window,
        absorber_rate=rate,
        on_axis=np.asarray(space.spectral_weights / (2 * np.pi), dtype=np.complex128),
        forward_scale=hankel.forward_scale,
        inverse_scale=hankel.inverse_scale,
    )
    return Transform(
        synthesis=synthesis,
        projection=projection,
        projection_coarse=projection,
        taper_scale=None,
        node_chunk=_node_chunk(space.n_r, n_pol, time.n_to, node_budget),
        aeff=tabulate(
            lambda z: np.full_like(z, np.pi * space.radius**2),
            0.0,
            z_max * (1 + margin),
        ),
        geometry=geometry,
        kind="radial",
    )


def radial_operators(  # noqa: PLR0913 - material and the shared translation controls
    space: RadialGrid,
    dispersion: SpatialDispersion,
    *,
    n_pol: int,
    z_max: float,
    absorber: Absorber | None = None,
    margin: float = DEFAULT_MARGIN,
    tol: float = DEFAULT_TOL,
    reuse_linear: ParameterExpansion | None = None,
    reuse_prefactor: ParameterExpansion | None = None,
) -> tuple[LinearOperator, PiecewiseChebyshev | ParameterExpansion]:
    """Build cumulative radial dispersion and the matching nonlinear prefactor.

    Parameters
    ----------
    space : solprop.grid.RadialGrid
        Radial spectral samples.
    dispersion : solprop.spatial.SpatialDispersion
        Bulk material, temporal grid and common frame.
    n_pol : int
        State polarisation components, 1 or 2.
    z_max : float
        Positive propagation length in m.
    absorber : solprop.boundaries.Absorber or None, optional
        Spectral power absorption rates in m^-1.
    margin : float, optional
        Relative extension of the cumulative tables beyond z_max.
    tol : float, optional
        Relative expansion target, positive. The linear expansion additionally
        budgets cancellation roundoff from the material and frame wave numbers.
    reuse_linear, reuse_prefactor : ParameterExpansion or None, optional
        Coefficients of unchanged operator functions on this grid, transverse
        geometry and frame. The caller verifies function identity; new paths must
        also lie inside the original sampled ranges to share their terms.

    Returns
    -------
    tuple
        Cumulative operator and prefactor representation, including temporal crop
        scaling. Constant media use exact constant tables; varying gas density uses
        the existing parameter expansions and cumulative scalar integral tables.

    Raises
    ------
    TranslationError
        If the run span is invalid or a varying medium cannot be resolved by an
        expansion, for example when it crosses a transverse cutoff.
    """
    span = z_max * (1 + margin)
    if not np.isfinite(span) or z_max <= 0 or margin < 0 or tol <= 0:
        msg = "radial operator requires a positive finite span and tolerance"
        raise TranslationError(msg)
    grid = dispersion.grid
    rate = np.zeros(grid.n_omega) if absorber is None else absorber.spectral_rate

    def sampled(
        model: SpatialDispersion, z: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Materialise radial coefficients with the complete seam shape.

        Parameters
        ----------
        model : solprop.spatial.SpatialDispersion
            Bulk material and frame at the sampled density.
        z : float, optional
            Position in m at which to evaluate the material.

        Returns
        -------
        tuple of numpy.ndarray
            Linear rate and cropped nonlinear prefactor, shared by polarisation.
        """
        square, frame, numerator, clamped = model.shared_values(z)
        return separable_values(
            SeparableOperator(
                k_squared=square,
                kperp_sq=space.k_r**2,
                k_squared_low=np.zeros_like(square),
                kperp_sq_low=np.zeros(space.n_r),
                frame=frame,
                spectral_rate=rate,
                prefactor_numerator=numerator * grid.crop_scale,
                prefactor_clamped=clamped * grid.crop_scale,
                active=grid.active_mask,
                evanescent_rate=model.evanescent_rate,
                n_pol=n_pol,
                z_span=(0.0, span),
            )
        )

    paths = {
        name: path
        for name, path in dispersion.medium.parameter_paths.items()
        if not path.is_constant
    }
    kinks = (
        np.unique(np.concatenate([path.breakpoints for path in paths.values()]))
        if paths
        else np.empty(0)
    )
    beta0 = tabulate(
        lambda z: np.array([dispersion.frame_quantities(float(one))[0] for one in z]),
        0.0,
        span,
        kinks=kinks,
    )
    beta1 = tabulate(
        lambda z: np.array([dispersion.frame_quantities(float(one))[1] for one in z]),
        0.0,
        span,
        kinks=kinks,
    )

    def at_parameters(values: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a frozen density sample for the parameter expansion.

        Parameters
        ----------
        values : collections.abc.Mapping
            Density sample in m^-3 under the name ``density``.

        Returns
        -------
        tuple of numpy.ndarray
            Linear rate and nonlinear prefactor at that density.
        """
        medium = dispersion.medium.at_parameters(values)
        return sampled(replace(dispersion, medium=medium))

    if not paths and reuse_linear is None and reuse_prefactor is None:
        operator, prefactor = sampled(dispersion)
        coefficients = np.ascontiguousarray(prefactor[None, None])
        edges = np.array([0.0, span])
        degrees = np.zeros(1, dtype=np.intp)
        kinks = np.empty(0)
        for values in (coefficients, edges, degrees, kinks):
            values.setflags(write=False)
        return (
            constant_phi(operator, z_max, margin=margin, beta0=beta0, beta1=beta1),
            PiecewiseChebyshev(coefficients, edges, degrees, kinks),
        )

    working = _linear_roundoff_target(dispersion, space, sampled, span, tol)
    if len(paths) > MAX_PARAMETERS:
        linear_table = build(
            lambda positions: np.stack(
                [sampled(dispersion, float(z))[0] for z in positions]
            ),
            0.0,
            span,
            tol=working,
            mandatory_breakpoints=kinks,
        )
        prefactor_table = build(
            lambda positions: np.stack(
                [sampled(dispersion, float(z))[1] for z in positions]
            ),
            0.0,
            span,
            tol=tol,
            mandatory_breakpoints=kinks,
        )
        if linear_table.budget_exceeded or prefactor_table.budget_exceeded:
            raise TranslationError(
                "independent radial material tables exceeded their interval budget"
            )
        # Frozen, as every array crossing the seam is: the problem's validation
        # refuses a writeable table.
        for values in (
            prefactor_table.coefficients,
            prefactor_table.breakpoints,
            prefactor_table.degrees,
            prefactor_table.hard_kinks,
        ):
            values.setflags(write=False)
        phi = linear_table.integral()
        return LinearOperator(
            phi_representation="table",
            phi_breakpoints=phi.breakpoints,
            phi_coefficients=phi.coefficients,
            phi_degrees=phi.degrees,
            hard_kinks=phi.hard_kinks,
            phi_constant=None,
            beta0=beta0,
            beta1=beta1,
        ), prefactor_table
    all_paths = dispersion.medium.parameter_paths
    linear = (
        None
        if reuse_linear is None
        else reweight_expansion(
            reuse_linear, all_paths, kinks=kinks, tol=working, integrate=True
        )
    )
    prefactor = (
        None
        if reuse_prefactor is None
        else reweight_expansion(reuse_prefactor, all_paths, kinks=kinks, tol=tol)
    )
    if not paths and (linear is None or prefactor is None):
        return radial_operators(
            space,
            dispersion,
            n_pol=n_pol,
            z_max=z_max,
            absorber=absorber,
            margin=margin,
            tol=tol,
        )
    if linear is None:
        linear = build_expansion(
            lambda values: at_parameters(values)[0],
            paths,
            0.0,
            span,
            kinks=kinks,
            tol=working,
            integrate=True,
        )
    if prefactor is None:
        prefactor = build_expansion(
            lambda values: at_parameters(values)[1],
            paths,
            0.0,
            span,
            kinks=kinks,
            tol=tol,
        )
    if linear is None or prefactor is None:
        msg = (
            "radial density expansion could not resolve the operator and prefactor; "
            "check whether the material crosses a transverse cutoff"
        )
        raise TranslationError(msg)
    logger.debug(
        "radial operator: density expansion, target %.3e, %d terms",
        working,
        linear.n_terms,
    )
    return LinearOperator(
        phi_representation="expansion",
        phi_breakpoints=linear.weights.breakpoints,
        phi_coefficients=None,
        phi_degrees=linear.weights.degrees,
        hard_kinks=linear.weights.hard_kinks,
        phi_constant=None,
        phi_expansion=linear,
        beta0=beta0,
        beta1=beta1,
    ), prefactor


def radial_rate_bounds(linear: LinearOperator) -> RateBounds:
    """Bound the real part of a radial operator over its whole span, for the budget.

    SEP 22 section 4.5 on the radial kind (its Q16). The radial operator is dense in
    ``(k_r, omega)`` rather than separable, and every backend forms the inverse stage
    factor ``exp(-dPhi)`` from it exactly as on the separable kinds, so a decaying
    column overflows it once ``h max(-Re D)`` passes ``log(DBL_MAX)``. A constant
    medium's bound is exact: ``dPhi = D h``, and the extremes are those of
    ``phi_constant``. A varying medium's rate is a piecewise Chebyshev series, z table
    or scalar weights of an expansion's fixed terms, and ``|T_j| <= 1`` on each
    interval bounds it: ``Re D`` of a z table lies within ``Re c_0 +- sum |Re c_j|``,
    and an expansion's weights within ``w_0 +- sum |w_j|``, combined with the sign of
    each term's real part. Both bounds are sufficient, never smaller than the rate the
    backends evaluate.

    The rate's parts (medium, capped evanescent attenuation, spectral rate) are not
    separable from ``D``, so the whole loss is reported as `propagating` and the other
    terms are zero.

    Parameters
    ----------
    linear : solprop.translate.linop.LinearOperator
        A radial operator, any representation but ``"separable"``.

    Returns
    -------
    solprop.translate.spatial.RateBounds
        ``loss`` bounds ``max(-Re D)`` and ``gain`` bounds ``max(Re D)``, m^-1.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.linop import constant_phi
    >>> from solprop.translate.radial import radial_rate_bounds
    >>> bounds = radial_rate_bounds(constant_phi(np.array([-3.0 - 1j, 0.5j]), 1.0))
    >>> bounds.loss, bounds.gain
    (3.0, 0.0)
    """
    if linear.phi_constant is not None:
        real = np.real(linear.phi_constant)
        lowest = float(np.min(real))
        highest = float(np.max(real))
    else:
        rate = linear.rate
        coefficients = rate.coefficients
        if linear.phi_expansion is None:
            centre = np.real(coefficients[:, 0])
            spread = np.sum(np.abs(np.real(coefficients[:, 1:])), axis=1)
            lowest = float(np.min(centre - spread))
            highest = float(np.max(centre + spread))
        else:
            weights = np.real(coefficients)
            spread = np.sum(np.abs(weights[:, 1:]), axis=1)
            low = weights[:, 0] - spread
            high = weights[:, 0] + spread
            terms = np.real(linear.phi_expansion.terms)
            terms = terms.reshape(terms.shape[0], -1)
            lowest, highest = np.inf, -np.inf
            for below, above in zip(low, high, strict=True):
                upper = np.maximum(below[:, None] * terms, above[:, None] * terms)
                lower = np.minimum(below[:, None] * terms, above[:, None] * terms)
                highest = max(highest, float(np.max(np.sum(upper, axis=0))))
                lowest = min(lowest, float(np.min(np.sum(lower, axis=0))))
    loss = max(-lowest, 0.0)
    return RateBounds(
        loss=loss,
        gain=max(highest, 0.0),
        propagating=loss,
        evanescent=0.0,
        spectral=0.0,
        transverse=0.0,
    )


def _linear_roundoff_target(
    dispersion: SpatialDispersion,
    space: RadialGrid,
    sampled: Callable[[SpatialDispersion, float], tuple[np.ndarray, np.ndarray]],
    span: float,
    requested: float,
) -> float:
    """Budget sqrt conditioning and frame subtraction before expansion sampling.

    Parameters
    ----------
    dispersion : solprop.spatial.SpatialDispersion
        Material and temporal frame.
    space : solprop.grid.RadialGrid
        Transverse frequencies for the conditioning calculation.
    sampled : callable
        Return the shaped operator and prefactor for a model and position.
    span : float
        Extended propagation interval in m.
    requested : float
        Requested dimensionless expansion tolerance, positive.

    Returns
    -------
    float
        Relative expansion target including the derived arithmetic floor.
    """
    # For s=sqrt(k^2-kperp^2), a relative perturbation in k^2 gives absolute
    # error |k^2|*epsilon/(2|s|). The following gamma(64) envelope covers both
    # material evaluations, square root and frame arithmetic. Eight such budgets
    # separate resolvable coefficients from arithmetic noise. The denominator is
    # exactly the expansion builder's per-element envelope convention.
    epsilon = np.finfo(float).eps
    rounding = 64 * epsilon / (1 - 64 * epsilon)
    peaks = np.zeros((space.n_r, dispersion.grid.n_omega))
    errors = np.zeros_like(peaks)
    for z in (0.0, span / 2, span):
        square, frame, _, _ = dispersion.shared_values(z)
        root = np.sqrt(square[None, :] - space.k_r[:, None] ** 2)
        absolute = np.abs(root)
        conditioning = np.zeros_like(absolute)
        np.divide(np.abs(square), 2 * absolute, out=conditioning, where=absolute > 0)
        errors = np.maximum(
            errors, rounding * (conditioning + absolute + np.abs(frame))
        )
        operator, _ = sampled(dispersion, z)
        peaks = np.maximum(peaks, np.abs(operator[0, :, 0]))
    denominator = np.maximum(peaks, ENVELOPE_FLOOR * float(peaks.max()))
    active = dispersion.grid.active_mask[None, :] & (denominator > 0)
    relative = np.zeros_like(peaks)
    np.divide(errors, denominator, out=relative, where=active)
    return max(requested, 8 * float(relative.max()))

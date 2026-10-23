# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Translate radial Hankel geometry and bulk propagation into the spatial seam."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
from jaxtyping import Complex

from solprop.boundaries import TEMPORAL_ALPHA_LENGTH, Absorber
from solprop.exceptions import TranslationError
from solprop.grid import RadialGrid, TimeGrid
from solprop.profiles import ZProfile
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
    endpoint_phi,
    tabulate,
)
from solprop.translate.modal import DEFAULT_NODE_BUDGET, Transform, _node_chunk
from solprop.translate.spatial import (
    RateBounds,
    SeparableOperator,
    SpatialGeometry,
    longitudinal_root,
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

    def material(model: SpatialDispersion, z: float = 0.0) -> SeparableOperator:
        """Assemble the radial coefficients of one material sample.

        Parameters
        ----------
        model : solprop.spatial.SpatialDispersion
            Bulk material and frame at the sampled density.
        z : float, optional
            Position in m at which to evaluate the material.

        Returns
        -------
        solprop.translate.spatial.SeparableOperator
            Shared material vectors over the radial columns.
        """
        square, frame, numerator, clamped = model.shared_values(z)
        return SeparableOperator(
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
        return separable_values(material(model, z))

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
    adiabatic = _AdiabaticAmplitude(
        dispersion,
        lambda model, z: longitudinal_root(material(model, z))[:2],
        paths,
        kinks,
        span,
        n_pol,
    )
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
        return adiabatic.add(
            LinearOperator(
                phi_representation="table",
                phi_breakpoints=phi.breakpoints,
                phi_coefficients=phi.coefficients,
                phi_degrees=phi.degrees,
                hard_kinks=phi.hard_kinks,
                phi_constant=None,
                beta0=beta0,
                beta1=beta1,
            )
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
    return adiabatic.add(
        LinearOperator(
            phi_representation="expansion",
            phi_breakpoints=linear.weights.breakpoints,
            phi_coefficients=None,
            phi_degrees=linear.weights.degrees,
            hard_kinks=linear.weights.hard_kinks,
            phi_constant=None,
            phi_expansion=linear,
            beta0=beta0,
            beta1=beta1,
        )
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

    A varying medium's rate includes the adiabatic amplitude term
    ``-1/2 d ln(Re k_z)/dz`` (SEP 34 section 4.4), which is positive where ``Re k_z``
    falls: its endpoint block is one more set of terms of the expansion, or part of
    the z table, so both bounds above include it and a lossless decreasing medium
    reports a gain.

    The rate's parts (medium, capped evanescent attenuation, spectral rate, adiabatic
    amplitude) are not separable from ``D``, so the whole loss is reported as
    `propagating` and the other terms are zero.

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


class _AdiabaticAmplitude:
    """The radial adiabatic amplitude term of the cumulative operator.

    SEP 34 section 4.4: in the internal convention ``Phi(z1) - Phi(z0)`` carries the
    real term ``-1/2 ln(Y_z(z1)/Y_z(z0))`` with ``Y_z = eps0 c^2 Re k_z / omega``,
    which per radial column and frequency is ``-1/2 ln(Re k_z(z1)/Re k_z(z0))``.
    It is applied on the active bins of the columns that propagate at the entrance
    and is exactly zero elsewhere; a column whose propagating status changes along
    the path is refused. The function resolved is the stable log-ratio against the
    entrance root, which vanishes there and is an endpoint quantity, so it is never
    obtained by differentiating a table (:func:`solprop.translate.linop.endpoint_phi`).

    Parameters
    ----------
    dispersion : solprop.spatial.SpatialDispersion
        Material along the run path.
    roots : callable
        ``(model, z) -> (k_z, propagating)``, each ``(n_r, n_omega)``, from the one
        root evaluation the operator uses
        (:func:`solprop.translate.spatial.longitudinal_root`).
    paths : collections.abc.Mapping
        The medium's varying parameter paths, by name.
    kinks : numpy.ndarray
        Breakpoints of those paths, m.
    span : float
        End of the tabulated range, m.
    n_pol : int
        State polarisation components, 1 or 2.

    Raises
    ------
    TranslationError
        If a column's propagating status changes along the sampled path.
    """

    def __init__(
        self,
        dispersion: SpatialDispersion,
        roots: Callable[[SpatialDispersion, float], tuple[np.ndarray, np.ndarray]],
        paths: Mapping[str, ZProfile],
        kinks: np.ndarray,
        span: float,
        n_pol: int,
    ) -> None:
        self._dispersion = dispersion
        self._roots = roots
        self._paths = paths
        self._n_pol = n_pol
        root, propagating = roots(dispersion, 0.0)
        self._mask = propagating & dispersion.grid.active_mask[None, :]
        self._reference = np.where(self._mask, np.real(root), 1.0)
        # The adiabatic certificate (solprop.adiabatic) has already refused a path
        # crossing a cutoff; every sample the builders take is checked again below.
        del kinks, span

    def _admitted(self, model: SpatialDispersion, z: float) -> np.ndarray:
        """Return ``Re k_z`` where the term applies, refusing a cutoff crossing.

        Parameters
        ----------
        model : solprop.spatial.SpatialDispersion
            Material sample.
        z : float
            Position in m at which to evaluate it.

        Returns
        -------
        numpy.ndarray
            ``Re k_z`` in m^-1, shape ``(n_r, n_omega)``; one where the term is zero.

        Raises
        ------
        TranslationError
            If the propagating set differs from the entrance's.
        """
        root, propagating = self._roots(model, z)
        changed = (propagating & self._dispersion.grid.active_mask[None, :]) != (
            self._mask
        )
        if np.any(changed):
            column, frequency = np.argwhere(changed)[0]
            msg = (
                f"radial column {int(column)} at omega = "
                f"{float(self._dispersion.grid.omega[frequency]):.6e} rad/s crosses "
                f"the transverse cutoff (q = 0) near z = {z:.6e} m; the adiabatic "
                "amplitude transport of a varying medium is refused across a cutoff"
            )
            raise TranslationError(msg)
        return np.where(self._mask, np.real(root), 1.0)

    def _log_ratio(
        self, model: SpatialDispersion, z: float
    ) -> Complex[np.ndarray, "1 n_r n_pol n_omega"]:
        """Return ``-1/2 ln(Re k_z / Re k_z(0))`` in the canonical layout.

        Parameters
        ----------
        model : solprop.spatial.SpatialDispersion
            Material sample.
        z : float
            Position in m at which to evaluate it.

        Returns
        -------
        numpy.ndarray
            Dimensionless, complex with zero imaginary part.
        """
        wave = self._admitted(model, z)
        reference = self._reference
        # Within a factor two of the reference the difference is exact (Sterbenz), so
        # log1p returns the small ratio to a few rounding units of itself rather than
        # to an absolute unit of one.
        near = (wave >= 0.5 * reference) & (wave <= 2.0 * reference)
        ratio = np.where(
            near,
            np.log1p((wave - reference) / reference),
            np.log(wave / reference),
        )
        values = np.where(self._mask, -0.5 * ratio, 0.0)
        shape = (1, values.shape[0], self._n_pol, values.shape[1])
        return np.broadcast_to(values[None, :, None, :], shape).astype(np.complex128)

    def sample(self, z: float) -> Complex[np.ndarray, "1 n_r n_pol n_omega"]:
        """Evaluate the term's endpoint function along the run path.

        Parameters
        ----------
        z : float
            Position in m.

        Returns
        -------
        numpy.ndarray
            Dimensionless endpoint function, canonical layout.
        """
        return self._log_ratio(self._dispersion, z)

    def value_at(
        self, values: Mapping[str, float]
    ) -> Complex[np.ndarray, "1 n_r n_pol n_omega"]:
        """Evaluate the endpoint function at one point of the parameter box.

        Parameters
        ----------
        values : collections.abc.Mapping
            Parameter values by name, inside the expansion's box.

        Returns
        -------
        numpy.ndarray
            Dimensionless endpoint function, canonical layout.
        """
        medium = self._dispersion.medium.at_parameters(values)
        return self._log_ratio(replace(self._dispersion, medium=medium), 0.0)

    def add(self, linear: LinearOperator) -> LinearOperator:
        """Join the term to a z table or an expansion of ``Phi``.

        Parameters
        ----------
        linear : solprop.translate.linop.LinearOperator
            The cumulative dispersion operator.

        Returns
        -------
        solprop.translate.linop.LinearOperator
            The operator with the endpoint term; unchanged when no column propagates
            or an expansion already carries it (batch reuse).
        """
        if not self._mask.any():
            return linear
        return endpoint_phi(
            linear,
            self.sample,
            self.value_at,
            self._paths,
            quantity="radial adiabatic amplitude",
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

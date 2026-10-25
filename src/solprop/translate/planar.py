# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Translate the hybrid planar grid into the kernel schema 15 ``"slab"`` seam.

The hybrid planar transform expands the field in the slab modes
of one gap in x and a raw DFT in y,

``E(x, y, t) = sum_p V_p(x) A_p(y, t)``,

so the state's columns are ``(p, k_y)`` with the mode index fastest. It is the
Cartesian ``"free"`` kind with its x FFT replaced by the slab rule's modal synthesis
and projection, and ``k_x**2`` replaced by each mode's complex, frequency-dependent
transverse square ``kappa_p(omega)**2``: the longitudinal wave number of a column is

``k_z = sqrt(k**2 eps_co - kappa_p**2 - k_y**2)``,

exact (non-paraxial) in y, with the free-space evanescent model and prefactor
``-i omega**2 / (2 eps0 c**2 Re k_z)`` per column. Constant media and a constant gap
only: both are refused by name upstream, and the operator here is frozen along z.

The kernel formula, the ``kappa_sq`` leaf and the geometry are
``docs/dev/kernel-contract.md`` sections 3 and 4; the physics is
``docs/guide/waveguides/planar.md``, "Resolving y".
"""

import warnings
from fractions import Fraction
from typing import cast

import numpy as np
from jaxtyping import Complex

from solprop.boundaries import TEMPORAL_ALPHA_LENGTH, Absorber
from solprop.constants import EPS0, C
from solprop.exceptions import GrazingIncidenceWarning, TranslationError
from solprop.grid import TimeGrid
from solprop.linear import frame_origin, frame_quantities
from solprop.modes.slab import PlanarGrid, SlabMode
from solprop.physdata import Gas, Mixture, as_gas
from solprop.physdata.glasses import GlassIndex
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.translate.linop import DEFAULT_MARGIN, LinearOperator, tabulate
from solprop.translate.modal import DEFAULT_SPATIAL_NODE_BUDGET, Transform
from solprop.translate.spatial import SeparableOperator, SpatialGeometry

__all__ = ["planar_kappa", "planar_operators", "planar_transform", "square_residuals"]


def _validate_span(n_pol: int, z_max: float, margin: float) -> float:
    """Validate the scalar state and a finite positive translation extent.

    Parameters
    ----------
    n_pol : int
        State polarisation components; the planar state carries one.
    z_max : float
        Run length in m.
    margin : float
        Fractional nonnegative extension.

    Returns
    -------
    float
        Extended positive span in m.

    Raises
    ------
    TranslationError
        If the state has two components or the span is invalid.
    """
    if isinstance(n_pol, bool) or n_pol != 1:
        msg = (
            "the hybrid planar transform carries a scalar state, n_pol = 1: its "
            "slab modes are one polarisation family, E_y for TE and E_x for TM"
        )
        raise TranslationError(msg)
    span = z_max * (1 + margin)
    if not np.isfinite(span) or z_max <= 0 or margin < 0:
        msg = "the planar operator requires a positive finite span"
        raise TranslationError(msg)
    return span


def planar_transform(  # noqa: PLR0913 - geometry and shared translation controls
    space: PlanarGrid,
    time: TimeGrid,
    *,
    n_pol: int,
    z_max: float,
    absorber: Absorber | None = None,
    node_budget: int = DEFAULT_SPATIAL_NODE_BUDGET,
    margin: float = DEFAULT_MARGIN,
) -> Transform:
    """Freeze the slab synthesis and projection in x and the y FFT layout.

    Parameters
    ----------
    space : solprop.modes.slab.PlanarGrid
        Slab modes and rule in x, periodic y axis and its collar.
    time : solprop.grid.TimeGrid
        Temporal grid: the pad scale folded into the synthesis, and the
        physical-node buffer size.
    n_pol : int
        State polarisation count; must be 1.
    z_max : float
        Positive run length in m.
    absorber : solprop.boundaries.Absorber or None, optional
        Spatial absorption reference length; None disables the in-stage y collar.
    node_budget : int, optional
        Shared node-buffer budget in bytes, positive.
    margin : float, optional
        Relative extension of diagnostic tables beyond z_max, nonnegative.

    Returns
    -------
    solprop.translate.modal.Transform
        The slab matrices in x, with the pad scale folded into the synthesis, the
        ``"slab"`` geometry and a bounded physical-node chunk.

    Raises
    ------
    TranslationError
        If the state has two components, the span is invalid or one temporal
        column does not fit the budget.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.modes import PlanarGrid, SlabMode
    >>> from solprop.translate.planar import planar_transform
    >>> grid = EnvelopeGrid.from_limits(0.1, 800e-9, (650e-9, 1000e-9), 400e-15)
    >>> space = PlanarGrid([SlabMode(63.5e-6, "Ar", 1.0)], 1e-3, 16)
    >>> transform = planar_transform(space, grid, n_pol=1, z_max=0.1)
    >>> transform.geometry.kind, transform.n_modes, transform.n_nodes
    ('slab', 16, 528)
    """
    span = _validate_span(n_pol, z_max, margin)
    # The same per-column accounting as the Cartesian transform: every physical
    # node owns one complete temporal history in the response pass.
    column_bytes = 20 * 16 * n_pol * time.n_to
    if node_budget < column_bytes:
        msg = "planar node_budget cannot hold one complete temporal column"
        raise TranslationError(msg)
    n_x = int(space.x.size)
    window = np.repeat(space.window_y, n_x)
    spatial_weights = np.tile(space.x_weights, space.n_y) * space.dy
    geometry = SpatialGeometry(
        kind="slab",
        shape=(n_x, space.n_y),
        axes=(space.x, space.y),
        spectral_axes=(space.k_y,),
        spatial_weights=spatial_weights,
        # A column is a modal amplitude in sqrt(W/m), not a field: dividing the
        # raw DFT's Parseval weight by eps0 c / 2 lets every backend keep its one
        # energy formula, eps0 c / 2 times the weighted spectral sum.
        spectral_weights=space.spectral_weights / (0.5 * EPS0 * C),
        coordinates=np.column_stack(
            (np.tile(space.x, space.n_y), np.repeat(space.y, n_x))
        ),
        window=window,
        absorber_rate=np.zeros(space.n_nodes)
        if absorber is None or not absorber.spatial
        else TEMPORAL_ALPHA_LENGTH * (1 - window) / absorber.in_stage_length,
        on_axis=space.on_axis,
        forward_scale=1.0,
        inverse_scale=1 / space.n_y,
    )
    return Transform(
        synthesis=space.synthesis * time.pad_scale,
        projection=space.projection,
        projection_coarse=space.projection_coarse,
        taper_scale=None,
        node_chunk=min(space.n_nodes, node_budget // column_bytes),
        aeff=tabulate(
            lambda z: np.full_like(z, np.sum(spatial_weights)),
            0.0,
            span,
        ),
        geometry=geometry,
        kind="slab",
    )


def planar_operators(
    space: PlanarGrid,
    dispersion: SpatialDispersion,
    *,
    n_pol: int,
    z_max: float,
    absorber: Absorber | None = None,
    margin: float = DEFAULT_MARGIN,
) -> tuple[LinearOperator, SeparableOperator]:
    """Keep the planar operator in separable form, with each channel's wall term.

    The material square ``k**2 eps_co`` and ``k_y**2`` are split into a rounded
    value and an exact residual as in the Cartesian operator; each channel's
    ``kappa_p**2 = k**2 eps_co - beta_p**2`` is formed from the mode's own complex
    index as ``(omega/c)**2 (n_co - n_p)(n_co + n_p)``, whose difference factor is
    exact for indices this close, so the column square at ``k_y = 0`` reproduces
    ``beta_p**2`` to a few rounding units of ``kappa_p**2``. The frame is the first
    channel's at ``k_y = 0``, as the modal route's, so a field uniform in y
    propagates exactly as the slab set does through ``prop_waveguide``.

    Parameters
    ----------
    space : solprop.modes.slab.PlanarGrid
        Slab channels and the y axis.
    dispersion : solprop.spatial.SpatialDispersion
        The fill as a bulk medium, the temporal grid and the frame.
    n_pol : int
        State polarisation count; must be 1.
    z_max : float
        Positive propagation length in m.
    absorber : solprop.boundaries.Absorber or None, optional
        Shared spectral absorption rate in m^-1.
    margin : float, optional
        Relative extension of the valid interval, nonnegative.

    Returns
    -------
    tuple
        Linear operator and the same separable record for the nonlinear prefactor.

    Raises
    ------
    TranslationError
        If the medium varies along z or is not the modes' own fill, the state has
        two components or the interval is invalid.

    Warns
    -----
    solprop.exceptions.GrazingIncidenceWarning
        If an active column lies within the arithmetic uncertainty of its cutoff.
    """
    if not all(path.is_constant for path in dispersion.medium.parameter_paths.values()):
        msg = (
            "the hybrid planar transform requires constant media: a gas path along "
            "z is refused, as for Cartesian free space"
        )
        raise TranslationError(msg)
    _check_medium(space, dispersion.medium)
    span = _validate_span(n_pol, z_max, margin)
    grid = dispersion.grid
    _, _, numerator, clamped = dispersion.shared_values()
    square, square_low = dispersion.material_squares()
    kappa = planar_kappa(space, dispersion)
    beta0, beta1 = frame_quantities(space.modes[0], dispersion.frame)
    origin = frame_origin(dispersion.frame, grid.kind)
    frame = beta0 + beta1 * (grid.omega - origin)
    data = SeparableOperator(
        k_squared=square,
        kperp_sq=space.kperp_sq,
        k_squared_low=square_low,
        kperp_sq_low=np.repeat(square_residuals(space.k_y), space.n_p),
        frame=frame,
        spectral_rate=np.zeros(grid.n_omega)
        if absorber is None
        else absorber.spectral_rate,
        prefactor_numerator=numerator * grid.crop_scale,
        prefactor_clamped=clamped * grid.crop_scale,
        active=grid.active_mask,
        evanescent_rate=dispersion.evanescent_rate,
        n_pol=n_pol,
        z_span=(0.0, span),
        kappa_sq=kappa,
    )
    _warn_grazing(space, grid, data)
    return LinearOperator(
        phi_representation="separable",
        phi_breakpoints=np.array([0.0, span]),
        phi_coefficients=None,
        phi_degrees=np.ones(1, dtype=np.intp),
        hard_kinks=np.empty(0),
        phi_constant=None,
        phi_expansion=None,
        beta0=tabulate(lambda z: np.full_like(z, beta0), 0.0, span),
        beta1=tabulate(lambda z: np.full_like(z, beta1), 0.0, span),
        separable=data,
    ), data


def planar_kappa(
    space: PlanarGrid, dispersion: SpatialDispersion
) -> Complex[np.ndarray, "n_p n_omega"]:
    """Return each slab channel's transverse wave-number square ``kappa_p**2``.

    ``kappa_p**2 = k**2 eps_co - beta_p**2`` is formed from the mode's own complex
    index as ``(omega/c)**2 (n_co - n_p)(n_co + n_p)``, whose difference factor is
    exact for indices this close.

    Parameters
    ----------
    space : solprop.modes.slab.PlanarGrid
        Slab channels and the y axis.
    dispersion : solprop.spatial.SpatialDispersion
        The fill as a bulk medium and the temporal grid.

    Returns
    -------
    numpy.ndarray
        Complex squares in m^-2, shape ``(n_p, n_omega)``, zero at inactive bins.
    """
    grid = dispersion.grid
    active = grid.active_mask
    omega = grid.omega[active]
    medium = dispersion.medium.index(omega)
    kappa = np.zeros((space.n_p, grid.n_omega), dtype=np.complex128)
    for row, mode in enumerate(space.modes):
        index = np.asarray(mode.neff(omega), dtype=np.complex128)
        kappa[row, active] = (omega / C) ** 2 * ((medium - index) * (medium + index))
    return kappa


def _check_medium(space: PlanarGrid, medium: SpatialMedium) -> None:
    """Refuse a bulk medium that is not the planar modes' own fill.

    The operator subtracts each mode's ``kappa_p**2 = k**2 eps_co - beta_p**2``, in
    which the medium's index cancels, but the nonlinear responses are scaled by the
    medium's density: a different medium would propagate correctly and respond as
    another gas. The modes carry the fill, so the medium must be that fill.

    Parameters
    ----------
    space : solprop.modes.slab.PlanarGrid
        The slab channels, which share one fill.
    medium : solprop.spatial.SpatialMedium
        The bulk medium the run was given.

    Raises
    ------
    TranslationError
        If the gas, its density (pressure and temperature), its susceptibility or
        the vacuum does not match the modes'.
    """
    mode = cast("SlabMode", space.modes[0])
    position = np.array([0.0])
    if mode.gas is None and mode.mixture is None:
        same = not isinstance(medium.material, Gas | Mixture | GlassIndex) and (
            complex(cast("complex", medium.material)) == 1.0
        )
    else:
        same = (
            mode.mixture is None
            and isinstance(medium.material, Gas)
            and medium.material == as_gas(cast("Gas | str", mode.gas))
            and medium.susceptibility is mode.susceptibility
            and float(medium.density(position)[0]) == float(mode.density(position)[0])
        )
    if not same:
        given = getattr(medium.material, "name", medium.material)
        msg = (
            "a planar grid's medium must be its slab modes' own fill, which sets the "
            f"density its nonlinear responses are scaled by: the modes carry {mode!r} "
            f"and the run was given {given} at "
            f"{float(medium.density(position)[0]):.6g} m^-3. Build the run with "
            "Simulation.from_planar or prop_planar, which take the fill from the modes"
        )
        raise TranslationError(msg)


def square_residuals(axis: np.ndarray) -> np.ndarray:
    """Return the exact residuals of an axis's rounded squares.

    Parameters
    ----------
    axis : numpy.ndarray
        Stored binary wave numbers in rad/m.

    Returns
    -------
    numpy.ndarray
        ``value**2`` in exact rational arithmetic minus its rounded float, in m^-2,
        one per entry, as the Cartesian ``kperp_sq_low`` is formed.
    """
    return np.array(
        [
            float(Fraction(float(value)) ** 2 - Fraction(float(value * value)))
            for value in axis
        ]
    )


def _warn_grazing(space: PlanarGrid, grid: TimeGrid, data: SeparableOperator) -> None:
    """Report the first active column inside its cutoff's arithmetic uncertainty.

    Parameters
    ----------
    space : solprop.modes.slab.PlanarGrid
        The channels and y frequencies, for the reported column.
    grid : solprop.grid.TimeGrid
        Frequency bins in rad/s and their active mask.
    data : solprop.translate.spatial.SeparableOperator
        Validated square parts and wall terms in m^-2.
    """
    assert data.kappa_sq is not None
    order = np.argsort(space.k_y**2, kind="stable")
    transverse = (space.k_y**2)[order]
    transverse_low = square_residuals(space.k_y)[order]
    epsilon = np.finfo(float).eps
    for omega_bin in np.flatnonzero(grid.active_mask):
        for channel in range(space.n_p):
            wall = data.kappa_sq[channel, omega_bin]
            material = data.k_squared[omega_bin]
            target = material.real - wall.real
            radius = 64 * epsilon * (abs(material) + abs(wall)) + 64 * np.nextafter(
                0.0, 1.0
            )
            first = np.searchsorted(transverse, target - radius, side="left")
            last = np.searchsorted(transverse, target + radius, side="right")
            if first == last:
                continue
            candidates = transverse[first:last]
            difference = (
                (material.real - candidates)
                + (data.k_squared_low[omega_bin].real - transverse_low[first:last])
            ) - wall.real
            # The wall term carries its own few rounding units, beside the
            # Cartesian split-square bound of the material and k_y parts.
            uncertainty = (
                8 * epsilon * np.abs(difference)
                + 64 * epsilon**2 * (abs(material) + candidates)
                + 8 * epsilon * abs(wall)
                + 32 * np.nextafter(0.0, 1.0)
            )
            unresolved = np.flatnonzero(np.abs(difference) <= uncertainty)
            if unresolved.size == 0:
                continue
            selected = int(unresolved[0])
            row = int(order[first + selected])
            msg = (
                "planar grazing-incidence rounding: active omega bin "
                f"{omega_bin} (omega={grid.omega[omega_bin]:.17g} rad/s), channel "
                f"{space.modes[channel].channel_label.display_label}, y row {row} has "
                f"Re(k^2 eps - kappa^2) - k_y^2 = {difference[selected]:.17g} m^-2 "
                f"within arithmetic uncertainty {uncertainty[selected]:.17g} m^-2. "
                "Roundoff can still change the clamp branch and the nonlinear "
                "1/Re(k_z) prefactor there; continuing with these coefficients."
            )
            warnings.warn(msg, GrazingIncidenceWarning, stacklevel=3)
            return

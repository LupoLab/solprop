# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Translate constant Cartesian propagation into the schema-5 separable seam."""

import warnings

import numpy as np
from jaxtyping import Float

from solprop.boundaries import TEMPORAL_ALPHA_LENGTH, Absorber
from solprop.exceptions import GrazingIncidenceWarning, TranslationError
from solprop.grid import FreeGrid, TimeGrid
from solprop.spatial import SpatialDispersion
from solprop.translate.linop import DEFAULT_MARGIN, LinearOperator, tabulate
from solprop.translate.modal import DEFAULT_SPATIAL_NODE_BUDGET, Transform
from solprop.translate.spatial import SeparableOperator, SpatialGeometry

__all__ = ["free_operators", "free_transform"]


def _validate_span(n_pol: int, z_max: float, margin: float) -> float:
    """Validate spatial polarisation and finite positive translation extent.

    Parameters
    ----------
    n_pol : int
        One or two independent Cartesian components.
    z_max : float
        Run length in m.
    margin : float
        Fractional nonnegative extension.

    Returns
    -------
    float
        Extended positive span in m.
    """
    if isinstance(n_pol, bool) or n_pol not in (1, 2):
        msg = "Cartesian polarisation count must be one or two"
        raise TranslationError(msg)
    span = z_max * (1 + margin)
    if not np.isfinite(span) or z_max <= 0 or margin < 0:
        msg = "Cartesian free operator requires a positive finite span"
        raise TranslationError(msg)
    return span


def _warn_grazing(space: FreeGrid, grid: TimeGrid, data: SeparableOperator) -> None:
    """Report the first active cutoff inside the split-square uncertainty bound.

    Parameters
    ----------
    space : solprop.grid.FreeGrid
        Cartesian shape for the reported column coordinates.
    grid : solprop.grid.TimeGrid
        Frequency bins in rad/s and their active mask.
    data : solprop.translate.spatial.SeparableOperator
        Validated square high/low parts in m^-2.
    """
    order = np.argsort(data.kperp_sq, kind="stable")
    transverse = data.kperp_sq[order]
    epsilon = np.finfo(float).eps
    # Search every high-part candidate, including ties with different residuals.
    # The first-order interval encloses all second-order-uncertain pairs without
    # constructing a spatial-frequency volume; see the predeclared contract.
    for omega_bin in np.flatnonzero(grid.active_mask):
        material = data.k_squared[omega_bin]
        radius = 64 * epsilon * abs(material) + 64 * np.nextafter(0.0, 1.0)
        first = np.searchsorted(transverse, material.real - radius, side="left")
        last = np.searchsorted(transverse, material.real + radius, side="right")
        columns = order[first:last]
        candidates = data.kperp_sq[columns]
        difference = (material.real - candidates) + (
            data.k_squared_low[omega_bin].real - data.kperp_sq_low[columns]
        )
        uncertainty = (
            8 * epsilon * np.abs(difference)
            + 64 * epsilon**2 * (abs(material) + candidates)
            + 32 * np.nextafter(0.0, 1.0)
        )
        unresolved = np.flatnonzero(np.abs(difference) <= uncertainty)
        if unresolved.size == 0:
            continue
        selected = int(unresolved[0])
        column = int(columns[selected])
        msg = (
            "Cartesian grazing-incidence rounding: active omega bin "
            f"{omega_bin} (omega={grid.omega[omega_bin]:.17g} rad/s), transverse "
            f"column {column} (ix={column % space.n_x}, iy={column // space.n_x}) "
            f"has Re(k²)-kperp²={difference[selected]:.17g} m^-2 "
            f"within arithmetic uncertainty {uncertainty[selected]:.17g} "
            "m^-2 after split-square correction. Roundoff can still change the "
            "clamp branch and nonlinear 1/Re(k_z) prefactor. Continuing with "
            "the corrected coefficients and existing nonpositive clamp; "
            "this grid diagnostic does not measure the affected field strength."
        )
        warnings.warn(msg, GrazingIncidenceWarning, stacklevel=3)
        return


def free_transform(  # noqa: PLR0913 - geometry and shared translation controls
    space: FreeGrid,
    time: TimeGrid,
    *,
    n_pol: int,
    z_max: float,
    absorber: Absorber | None = None,
    node_budget: int = DEFAULT_SPATIAL_NODE_BUDGET,
    margin: float = DEFAULT_MARGIN,
) -> Transform:
    """Freeze the Cartesian FFT layout and bounded physical-column chunk.

    Parameters
    ----------
    space : solprop.grid.FreeGrid
        Validated Cartesian axes, raw-DFT measures and spatial collar.
    time : solprop.grid.TimeGrid
        Temporal grid defining the physical-column buffer size.
    n_pol : int
        State polarisation count, 1 or 2.
    z_max : float
        Positive run length in m.
    absorber : solprop.boundaries.Absorber or None, optional
        Spatial absorption reference length; None disables spatial absorption.
    node_budget : int, optional
        Shared node-buffer budget in bytes, positive.
    margin : float, optional
        Relative extension of diagnostic tables beyond z_max, nonnegative.

    Returns
    -------
    solprop.translate.modal.Transform
        Empty dense matrices, explicit FFT geometry and chunk size.
    """
    space.validate()
    span = _validate_span(n_pol, z_max, margin)
    # Count complex-equivalent columns before allocation: eight response/FFT
    # buffers, three doubled Raman buffers, 3.5 device plasma buffers, two
    # split-step buffers and 2.5 statistic buffers. Their sum is 19; the extra
    # column covers real-FFT endpoint bins. The transverse scratch and opaque
    # FFT plan allocations are separately named in the memory ledger.
    column_bytes = 20 * 16 * n_pol * time.n_to
    if node_budget < column_bytes:
        msg = "Cartesian node_budget cannot hold one complete temporal column"
        raise TranslationError(msg)
    geometry = SpatialGeometry(
        kind="free",
        shape=(space.n_x, space.n_y),
        axes=(space.x, space.y),
        spectral_axes=(space.k_x, space.k_y),
        spatial_weights=space.spatial_weights,
        spectral_weights=space.spectral_weights,
        coordinates=np.column_stack(
            (np.tile(space.x, space.n_y), np.repeat(space.y, space.n_x))
        ),
        window=space.window,
        absorber_rate=np.zeros(space.n_columns)
        if absorber is None
        else TEMPORAL_ALPHA_LENGTH * (1 - space.window) / absorber.in_stage_length,
        on_axis=space.on_axis,
        forward_scale=1.0,
        inverse_scale=1 / space.n_columns,
    )
    return Transform(
        synthesis=np.empty((0, 0)),
        projection=np.empty((0, 0)),
        projection_coarse=np.empty((0, 0)),
        taper_scale=None,
        node_chunk=min(space.n_columns, node_budget // column_bytes),
        aeff=tabulate(
            lambda z: np.full_like(z, np.sum(space.spatial_weights)),
            0.0,
            span,
        ),
        geometry=geometry,
        kind="free",
    )


def free_operators(  # noqa: PLR0913 - geometry, operator settings and controls
    space: FreeGrid,
    dispersion: SpatialDispersion,
    *,
    n_pol: int,
    z_max: float,
    absorber: Absorber | None = None,
    margin: float = DEFAULT_MARGIN,
    transverse_rate: Float[np.ndarray, " n_modes"] | None = None,
) -> tuple[LinearOperator, SeparableOperator]:
    """Keep constant Cartesian material coefficients in separable form.

    Parameters
    ----------
    space : solprop.grid.FreeGrid
        Cartesian transverse wave-number squares in m^-2.
    dispersion : solprop.spatial.SpatialDispersion
        Constant bulk material, temporal grid and frame.
    n_pol : int
        State polarisation count, 1 or 2.
    z_max : float
        Positive propagation length in m.
    absorber : solprop.boundaries.Absorber or None, optional
        Shared spectral absorption rate in m^-1.
    margin : float, optional
        Relative extension of the valid interval, nonnegative.
    transverse_rate : numpy.ndarray or None, optional
        A transverse collar's power rate per column in m^-1
        (:func:`solprop.boundaries.transverse_collar_rate`); ``None`` is zero.

    Returns
    -------
    tuple
        Linear operator and the same separable record for the nonlinear prefactor.

    Raises
    ------
    TranslationError
        If the medium varies along z or the interval is invalid.

    Warns
    -----
    solprop.exceptions.GrazingIncidenceWarning
        If an active transverse/frequency bin lies within the elementary-arithmetic
        uncertainty of the grazing clamp branch. Translation continues unchanged;
        this diagnostic does not assess the field or nonlinear source in that bin.
    """
    if not all(path.is_constant for path in dispersion.medium.parameter_paths.values()):
        msg = (
            "Cartesian free propagation requires constant media; "
            "varying density profiles are unsupported"
        )
        raise TranslationError(msg)
    span = _validate_span(n_pol, z_max, margin)
    grid = dispersion.grid
    _, frame, numerator, clamped = dispersion.shared_values()
    square, square_low, transverse_low = dispersion.cartesian_squares(space)
    data = SeparableOperator(
        k_squared=square,
        kperp_sq=space.kperp_sq,
        k_squared_low=square_low,
        kperp_sq_low=transverse_low,
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
        kperp_rate=transverse_rate,
    )
    _warn_grazing(space, grid, data)
    beta0, beta1 = dispersion.frame_quantities(0.0)
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

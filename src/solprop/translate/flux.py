# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build the axial-admittance plan of one spatial plane (SEP 34).

A free-space field's power crossing a plane is measured with the scalar forward
equation's axial admittance ``Y_c = eps0 c**2 k_z / omega`` of the medium at that
plane (``docs/guide/free-space/flux.md``). Every evaluation reads the one
longitudinal root of :func:`solprop.translate.spatial.longitudinal_root`, on the
same separable coefficients the propagation uses, so a plane's measure and its
propagation see the same ``k_z``.
"""

import math

import numpy as np
from jaxtyping import Complex, Float

from solprop.constants import C
from solprop.exceptions import MaterialError
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.linear import Frame
from solprop.modes.slab import PlanarGrid
from solprop.spatial import EVANESCENT_RATE, SpatialDispersion, SpatialMedium
from solprop.translate.planar import planar_kappa, square_residuals
from solprop.translate.spatial import SeparableOperator, longitudinal_root

__all__ = ["axial_admittance", "plane_coefficients"]


def plane_coefficients(
    space: RadialGrid | FreeGrid | PlanarGrid,
    medium: SpatialMedium,
    grid: TimeGrid,
    z: float,
    *,
    evanescent_rate: float = EVANESCENT_RATE,
) -> SeparableOperator:
    """Return the separable coefficients of one plane's medium.

    The material square and the transverse squares are split exactly as the run's
    own operator splits them (radial: rounded squares; Cartesian and planar: a
    rounded square and its exact residual; planar: each channel's wall term), so
    the plane's ``k_z`` is the propagation's to rounding. The frame, rates and
    prefactors are not part of the measure and are left zero.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        Transverse geometry of the plane.
    medium : solprop.spatial.SpatialMedium
        The bulk medium; for a planar grid, the slab modes' own fill.
    grid : solprop.grid.TimeGrid
        Temporal grid; only its active band samples the material.
    z : float
        Finite position of the plane in m, at which the medium is evaluated.
    evanescent_rate : float, optional
        Evanescent attenuation cap in m^-1 carried for the root's rate.

    Returns
    -------
    SeparableOperator
        Coefficients with the plane's material and transverse squares.

    Raises
    ------
    MaterialError
        If the position is not finite.
    """
    if not math.isfinite(z):
        msg = "a flux plane needs a finite position"
        raise MaterialError(msg)
    # The frame does not enter the measure; a fixed zero frame skips its derivative.
    frame = Frame(grid.omega0, subtract_beta0=False, beta0=0.0, beta1=0.0)
    dispersion = SpatialDispersion(medium, grid, frame, evanescent_rate=evanescent_rate)
    zeros = np.zeros(grid.n_omega)
    kappa = None
    if isinstance(space, RadialGrid):
        square, _, _, _ = dispersion.shared_values(z)
        square_low = np.zeros_like(square)
        transverse, transverse_low = space.k_r**2, np.zeros(space.n_r)
    elif isinstance(space, FreeGrid):
        square, square_low, transverse_low = dispersion.cartesian_squares(space, z)
        transverse = space.kperp_sq
    else:
        square, square_low = dispersion.material_squares(z)
        transverse = space.kperp_sq
        transverse_low = np.repeat(square_residuals(space.k_y), space.n_p)
        kappa = planar_kappa(space, dispersion)
    return SeparableOperator(
        k_squared=square,
        kperp_sq=transverse,
        k_squared_low=square_low,
        kperp_sq_low=transverse_low,
        frame=zeros,
        spectral_rate=zeros,
        prefactor_numerator=zeros.astype(complex),
        prefactor_clamped=zeros.astype(complex),
        active=grid.active_mask,
        evanescent_rate=evanescent_rate,
        n_pol=1,
        z_span=(z - 1.0, z + 1.0),
        kappa_sq=kappa,
    )


def axial_admittance(
    coefficients: SeparableOperator,
    omega: Float[np.ndarray, " n_omega"],
    columns: slice,
) -> Complex[np.ndarray, "n_chunk n_omega"]:
    """Return the dimensionless forward admittance ``c k_z / omega`` of a chunk.

    The physical admittance is ``eps0 c`` times this value. It is the root of
    :func:`solprop.translate.spatial.longitudinal_root` on active propagating bins
    at positive frequency, in the optics convention (positive imaginary part for
    loss), and zero elsewhere: a nonpropagating or inactive column is not a forward
    power port. Its real part is the obliquity of the far-field statistics.

    Parameters
    ----------
    coefficients : SeparableOperator
        The plane's or the run's constant coefficients.
    omega : numpy.ndarray
        Physical angular frequencies in rad/s, shape ``(n_omega,)``.
    columns : slice
        The transverse columns of the chunk.

    Returns
    -------
    numpy.ndarray
        Complex admittance, shape ``(n_chunk, n_omega)``.
    """
    root, propagating, _ = longitudinal_root(coefficients, columns)
    live = propagating & coefficients.active[None, :] & (omega > 0)[None, :]
    admittance = np.zeros(root.shape, dtype=np.complex128)
    np.divide(C * root, omega[None, :], out=admittance, where=live)
    return admittance

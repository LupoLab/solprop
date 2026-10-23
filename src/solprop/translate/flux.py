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
from collections.abc import Mapping
from typing import cast

import numpy as np

from solprop.exceptions import MaterialError, TranslationError
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.linear import Frame
from solprop.modes.slab import PlanarGrid
from solprop.spatial import EVANESCENT_RATE, SpatialDispersion, SpatialMedium
from solprop.translate.chebyshev import PiecewiseChebyshev, build
from solprop.translate.linop import LinearOperator
from solprop.translate.planar import planar_kappa, square_residuals
from solprop.translate.spatial import FluxPlan, SeparableOperator

__all__ = ["flux_from_record", "flux_plan", "flux_record", "plane_coefficients"]


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


def flux_plan(
    space: RadialGrid | FreeGrid | PlanarGrid,
    dispersion: SpatialDispersion,
    linear: LinearOperator,
    *,
    tol: float,
) -> FluxPlan:
    """Build the run's plane-measure coefficients for its statistics.

    Cartesian, line and planar runs are uniform along z, so the run's own separable
    coefficients are the measure everywhere. A radial run's coefficients at the
    entrance are paired with a z table of the material square when its medium
    varies, built on the medium's breakpoints as mandatory edges.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        Transverse geometry.
    dispersion : solprop.spatial.SpatialDispersion
        The run's medium, temporal grid and evanescent cap.
    linear : solprop.translate.linop.LinearOperator
        The run's linear operator: its separable record and its span.
    tol : float
        Relative tolerance of the material table, positive.

    Returns
    -------
    FluxPlan
        The run's plane measure.
    """
    if linear.separable is not None:
        return FluxPlan(linear.separable)
    grid = dispersion.grid
    span = (float(linear.phi_breakpoints[0]), float(linear.phi_breakpoints[-1]))
    base = plane_coefficients(
        space, dispersion.medium, grid, 0.0, evanescent_rate=dispersion.evanescent_rate
    )
    paths = [
        path
        for path in dispersion.medium.parameter_paths.values()
        if not path.is_constant
    ]
    if not paths:
        return FluxPlan(base)
    kinks = np.unique(np.concatenate([path.breakpoints for path in paths]))
    material = build(
        lambda positions: np.stack(
            [dispersion.shared_values(float(one))[0] for one in positions]
        ),
        *span,
        tol=tol,
        mandatory_breakpoints=kinks[(kinks > span[0]) & (kinks < span[1])],
    )
    for values in (
        material.coefficients,
        material.breakpoints,
        material.degrees,
        material.hard_kinks,
    ):
        values.setflags(write=False)
    return FluxPlan(base, material)


def flux_record(plan: FluxPlan | None) -> dict[str, object]:
    """Return a plane measure as plain arrays for the output's spatial record.

    Parameters
    ----------
    plan : FluxPlan or None
        The run's plane measure.

    Returns
    -------
    dict
        The coefficients :func:`flux_from_record` rebuilds the measure from.

    Raises
    ------
    TranslationError
        If a spatial run carries no plane measure.
    """
    if plan is None:
        msg = "a spatial run's metadata needs its plane measure"
        raise TranslationError(msg)
    base = plan.base
    record: dict[str, object] = {
        "k_squared": base.k_squared,
        "k_squared_low": base.k_squared_low,
        "kperp_sq": base.kperp_sq,
        "kperp_sq_low": base.kperp_sq_low,
        "active": base.active,
        "evanescent_rate": base.evanescent_rate,
        "kappa_sq": base.kappa_sq,
        "material": None,
    }
    if plan.material is not None:
        table = plan.material
        record["material"] = {
            "coefficients": table.coefficients,
            "breakpoints": table.breakpoints,
            "degrees": table.degrees,
            "hard_kinks": table.hard_kinks,
        }
    return record


def flux_from_record(record: Mapping[str, object]) -> FluxPlan:
    """Rebuild a plane measure from its output record.

    Parameters
    ----------
    record : collections.abc.Mapping
        The ``flux_plan`` entry of a spatial output record.

    Returns
    -------
    FluxPlan
        The run's plane measure, whose :meth:`FluxPlan.at` gives a saved plane's
        coefficients at its own position.
    """

    def array(name: str, dtype: type) -> np.ndarray:
        """Return one stored coefficient array."""
        return np.asarray(record[name], dtype=dtype)

    zeros = np.zeros(array("k_squared", complex).size)
    kappa = record["kappa_sq"]
    base = SeparableOperator(
        k_squared=array("k_squared", complex),
        kperp_sq=array("kperp_sq", float),
        k_squared_low=array("k_squared_low", complex),
        kperp_sq_low=array("kperp_sq_low", float),
        frame=zeros,
        spectral_rate=zeros,
        prefactor_numerator=zeros.astype(complex),
        prefactor_clamped=zeros.astype(complex),
        active=array("active", bool),
        evanescent_rate=float(cast("float", record["evanescent_rate"])),
        n_pol=1,
        # A measure is evaluated where it is asked; its span is never stepped.
        z_span=(0.0, 1.0),
        kappa_sq=None if kappa is None else np.asarray(kappa, complex),
    )
    stored = record["material"]
    if stored is None:
        return FluxPlan(base)
    table = cast("Mapping[str, object]", stored)
    material = PiecewiseChebyshev(
        np.asarray(table["coefficients"], complex),
        np.asarray(table["breakpoints"], float),
        np.asarray(table["degrees"], np.intp),
        np.asarray(table["hard_kinks"], float),
    )
    return FluxPlan(base, material)

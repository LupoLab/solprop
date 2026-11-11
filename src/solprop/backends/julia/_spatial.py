# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own scalar spatial source descriptors without loading either compiled backend."""

from dataclasses import dataclass
from typing import Literal
from weakref import WeakKeyDictionary

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop.backends.julia._descriptors import (
    ModalTransfer,
    SourceTransfer,
    lower_mapped,
    lower_source,
    owned,
)
from solprop.exceptions import BackendError
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import SpatialRoute
from solprop.translate.spatial import SeparableOperator

__all__ = [
    "CartesianTransfer",
    "RadialTransfer",
    "SeparableTransfer",
    "lower_separable",
    "lower_spatial",
]


@dataclass(frozen=True, eq=False)
class SeparableTransfer:
    """Own the existing constant separated spatial coefficients without expanding them.

    Attributes
    ----------
    k_squared, k_squared_low : numpy.ndarray
        Complex material square and its residual, shape ``(n_omega,)``, m^-2.
    kperp_sq, kperp_sq_low : numpy.ndarray
        Transverse square and its residual, shape ``(n_modes,)``, m^-2.
    frame, spectral_rate : numpy.ndarray
        Moving-frame wave number and spectral power loss, inverse metres.
    prefactor_numerator, prefactor_clamped : numpy.ndarray
        Translated imaginary source numerator and nonpropagating clamp.
    active : numpy.ndarray
        Exact Boolean frequency mask.
    evanescent_rate : float
        Positive nonpropagating field-loss clamp, inverse metres.
    z_span : tuple of float
        Admitted finite increasing interval, metres.
    kperp_rate : numpy.ndarray or None
        Nonnegative power loss per transverse column, inverse metres.
    """

    k_squared: Complex[np.ndarray, " n_omega"]
    k_squared_low: Complex[np.ndarray, " n_omega"]
    kperp_sq: Float[np.ndarray, " n_modes"]
    kperp_sq_low: Float[np.ndarray, " n_modes"]
    frame: Float[np.ndarray, " n_omega"]
    spectral_rate: Float[np.ndarray, " n_omega"]
    prefactor_numerator: Complex[np.ndarray, " n_omega"]
    prefactor_clamped: Complex[np.ndarray, " n_omega"]
    active: Bool[np.ndarray, " n_omega"]
    evanescent_rate: float
    z_span: tuple[float, float]
    kperp_rate: Float[np.ndarray, " n_modes"] | None


@dataclass(frozen=True, eq=False)
class RadialTransfer:
    """Pair a scalar Hankel matrix preparation with its additive physical absorber.

    Attributes
    ----------
    modal : ModalTransfer
        Existing static radial maps and represented source and linear coefficients.
    spatial_rate : numpy.ndarray
        Nonnegative physical power loss, shape ``(n_nodes,)``, inverse metres.
    pad_scale : float
        Existing dimensionless fine temporal padding factor, applied once.
    kind : str
        Immutable bridge discriminator for the radial or Cartesian layout.
    """

    modal: ModalTransfer
    spatial_rate: Float[np.ndarray, " n_nodes"]
    pad_scale: float
    kind: Literal["radial"] = "radial"


@dataclass(frozen=True, eq=False)
class CartesianTransfer:
    """Pair scalar material data with an FFT shape and separated spectral coefficients.

    Attributes
    ----------
    source : SourceTransfer
        Fine temporal grid and independent material responses.
    operator : SeparableTransfer
        Shared linear and nonlinear separated coefficients.
    shape : tuple of int
        Physical ``(n_x,n_y)`` FFT shape, with x fastest in flattened columns.
    spatial_rate : numpy.ndarray
        Nonnegative physical power loss, shape ``(n_nodes,)``, inverse metres.
    pad_scale : float
        Existing dimensionless fine temporal padding factor, applied once.
    kind : str
        Immutable bridge discriminator for the radial or Cartesian layout.
    """

    source: SourceTransfer
    operator: SeparableTransfer
    shape: tuple[int, int]
    spatial_rate: Float[np.ndarray, " n_nodes"]
    pad_scale: float
    kind: Literal["cartesian"] = "cartesian"


_SPATIAL: WeakKeyDictionary[PropagationProblem, RadialTransfer | CartesianTransfer] = (
    WeakKeyDictionary()
)


def lower_separable(operator: SeparableOperator) -> SeparableTransfer:
    """Copy scalar separated coefficients exactly, including both residual words.

    Parameters
    ----------
    operator : solprop.translate.spatial.SeparableOperator
        Validated constant spatial coefficients in their translated SI units.

    Returns
    -------
    SeparableTransfer
        Independent read-only arrays without a frequency-by-transverse volume.

    Raises
    ------
    BackendError
        For vector or planar coefficients outside the accepted scalar route.
    """
    if operator.n_pol != 1 or operator.kappa_sq is not None:
        raise BackendError("julia spatial coefficients refuse vector and planar routes")
    return SeparableTransfer(
        owned(operator.k_squared),
        owned(operator.k_squared_low),
        owned(operator.kperp_sq),
        owned(operator.kperp_sq_low),
        owned(operator.frame),
        owned(operator.spectral_rate),
        owned(operator.prefactor_numerator),
        owned(operator.prefactor_clamped),
        owned(operator.active),
        operator.evanescent_rate,
        operator.z_span,
        None if operator.kperp_rate is None else owned(operator.kperp_rate),
    )


def lower_spatial(problem: PropagationProblem) -> RadialTransfer | CartesianTransfer:
    """Cache exact scalar radial, Cartesian or line preparation without runtime entry.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Validated scalar spatial descriptor.

    Returns
    -------
    RadialTransfer or CartesianTransfer
        Owned immutable inputs; no mutable workspace or original problem is retained.

    Raises
    ------
    BackendError
        Naming unsupported routes, responses or nonconstant Cartesian coefficients.
    """
    if problem in _SPATIAL:
        return _SPATIAL[problem]
    route, geometry = problem.route, problem.transform.geometry
    if not isinstance(route, SpatialRoute) or geometry is None:
        raise BackendError("julia spatial preparation requires a spatial route")
    if route.kind == "planar":
        raise BackendError("julia spatial preparation refuses the planar route")
    result: RadialTransfer | CartesianTransfer
    if route.kind == "radial":
        result = RadialTransfer(
            lower_mapped(problem),
            owned(geometry.absorber_rate),
            problem.grids.pad_scale,
        )
    else:
        operator = problem.linear.separable
        if operator is None:
            raise BackendError("julia Cartesian sources require constant coefficients")
        source = lower_source(problem)
        result = CartesianTransfer(
            source,
            lower_separable(operator),
            (geometry.shape[0], geometry.shape[1]),
            owned(geometry.absorber_rate),
            problem.grids.pad_scale,
        )
    _SPATIAL[problem] = result
    return result

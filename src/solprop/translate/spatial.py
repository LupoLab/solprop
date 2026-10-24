# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Freeze spatial geometry and separable coefficients for the kernel contract.

These records contain arrays and scalars only. Geometry is independent of the
temporal grid. A separable operator keeps frequency-only material coefficients and
one transverse-frequency square per column, avoiding a full coefficient volume.
Kernel schema 15 adds the hybrid planar kind ``"slab"`` (slab modes in x, an FFT in
y) and the one per-channel table it needs, ``kappa_sq``.
"""

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop.adiabatic import AdiabaticAdmission
from solprop.constants import C
from solprop.exceptions import OutputError, TranslationError
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.spatial import (
    FAR_FIELD_STATISTICS,
    ApertureTables,
    aperture_scale,
    attenuation_extremes,
    quadrature_half_width,
)
from solprop.stats import StatsSpec
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.masks import Grids

__all__ = [
    "FluxPlan",
    "RateBounds",
    "SeparableOperator",
    "SpatialGeometry",
    "aperture_tables",
    "axial_admittance",
    "axial_factors",
    "longitudinal_root",
    "rate_bounds",
    "separable_values",
]


@dataclass(frozen=True, eq=False)
class SpatialGeometry:
    """Carry spatial axes, quadrature, reconstruction and absorption across the seam.

    Attributes
    ----------
    kind : {'radial', 'free', 'slab'}
        Radial QDHT, Cartesian FFT, or the hybrid planar transform: slab modes in x
        (the transform's own synthesis and projection on the slab rule) times an FFT
        in y. Modal transforms carry no such record.
    shape : tuple of int
        ``(n_r,)`` or ``(n_x, n_y)``: the **physical** node grid. Cartesian column
        ``ix + n_x*iy`` has x fastest. Every entry is at least 2, except ``n_y = 1``
        for a line uniform in y, whose one y sample is ``y = k_y = 0`` with unit
        measure (per metre of y). For ``"slab"``, ``n_x`` counts the slab rule's x
        nodes and the state's columns are ``(p, k_y)``, ``p + n_p*iy`` with the mode
        index fastest, so ``n_modes = n_p*n_y`` while ``n_nodes = n_x*n_y``.
    axes : tuple of numpy.ndarray
        ``(r,)`` or ``(x, y)``, float64 in m, strictly increasing. A slab's x axis is
        its quadrature nodes inside the gap, not uniformly spaced.
    spectral_axes : tuple of numpy.ndarray
        ``(k_r,)`` or ``(k_x, k_y)``, in rad/m; for ``"slab"`` ``(k_y,)`` alone, the
        x direction's spectrum being the mode index. Cartesian axes are in FFT order.
    spatial_weights : numpy.ndarray
        Physical area weights in m^2, shape ``(n_nodes,)`` in node order.
    spectral_weights : numpy.ndarray
        Parseval weights for the stored transverse spectrum, shape ``(n_modes,)``.
        Units are m^-2 for radial Hankel spectra and m^2 for Cartesian raw DFTs. A
        slab column is a modal amplitude in sqrt(W/m) and its weight ``2 dy / (eps0
        c n_y)`` is the raw DFT's ``dy/n_y`` divided by ``eps0 c / 2``, in m ohm, so
        that the common ``eps0 c / 2`` energy measure gives joules on every kind.
    coordinates : numpy.ndarray
        ``(r, 0)`` or ``(x, y)`` per physical node, in m, shape ``(n_nodes, 2)``.
    window : numpy.ndarray
        Spatial collar window in [0, 1], shape ``(n_nodes,)``. For ``"slab"`` it
        is the y collar alone, the same at every x node of a row.
    absorber_rate : numpy.ndarray
        In-stage spatial power absorption rate, in 1/m, shape ``(n_nodes,)``; for
        ``"slab"`` a function of y alone, like `window`.
    on_axis : numpy.ndarray
        Linear reconstruction weights from transverse spectral columns to the axis,
        complex128, shape ``(n_modes,)``. Temporal oversampling is not folded in.
    forward_scale : float
        Common transverse forward scaling; m^2 radially, dimensionless for the FFT.
    inverse_scale : float
        Common transverse inverse scaling; m^-2 radially, dimensionless for the FFT
        (``1/n_y`` for the slab's y transform).

    Raises
    ------
    TranslationError
        If dimensions, axes, measures, collar values or scalings disagree.
    """

    kind: Literal["radial", "free", "slab"]
    shape: tuple[int, ...]
    axes: tuple[Float[np.ndarray, " ..."], ...]
    spectral_axes: tuple[Float[np.ndarray, " ..."], ...]
    spatial_weights: Float[np.ndarray, " n_area"]
    spectral_weights: Float[np.ndarray, " n_spectral_area"]
    coordinates: Float[np.ndarray, "n_points 2"]
    window: Float[np.ndarray, " n_window"]
    absorber_rate: Float[np.ndarray, " n_rate"]
    on_axis: Complex[np.ndarray, " n_axis_columns"]
    forward_scale: float
    inverse_scale: float

    def __post_init__(self) -> None:
        """Own immutable arrays and check their common spatial shape."""
        dimension = 1 if self.kind == "radial" else 2
        if self.kind not in ("radial", "free", "slab") or len(self.shape) != dimension:
            msg = (
                "spatial kind and shape must be radial (n_r,), free (n_x, n_y) or "
                "slab (n_x, n_y)"
            )
            raise TranslationError(msg)
        # A Cartesian line uniform in y (solprop.grid.LineGrid) is the one
        # exception to two samples per axis: n_y = 1, carrying only k_y = 0.
        if any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < (1 if axis == 1 and self.kind == "free" else 2)
            for axis, size in enumerate(self.shape)
        ):
            msg = (
                "spatial shape entries must be integers of at least 2, "
                "except a Cartesian line's n_y = 1"
            )
            raise TranslationError(msg)
        for name in ("axes", "spectral_axes"):
            axes = tuple(_frozen_array(one, np.float64) for one in getattr(self, name))
            # A slab's x spectrum is its mode index, carried by the transform's
            # matrices, so its only spectral axis is k_y.
            sizes = (
                self.shape[1:]
                if self.kind == "slab" and name == "spectral_axes"
                else self.shape
            )
            if len(axes) != len(sizes) or any(
                one.shape != (size,) for one, size in zip(axes, sizes, strict=True)
            ):
                msg = f"spatial {name} must match shape {self.shape}"
                raise TranslationError(msg)
            object.__setattr__(self, name, axes)
        for axis in self.axes:
            if np.any(np.diff(axis) <= 0):
                msg = "physical spatial axes must be strictly increasing"
                raise TranslationError(msg)
        self._adopt_vectors()
        if self.kind == "slab":
            self._check_slab_columns()
        self._check_coordinates()
        if np.any(self.spatial_weights <= 0) or np.any(self.spectral_weights <= 0):
            msg = "spatial and spectral integration weights must be positive"
            raise TranslationError(msg)
        if np.any((self.window < 0) | (self.window > 1)) or np.any(
            self.absorber_rate < 0
        ):
            msg = "spatial window must be in [0, 1] and absorber_rate nonnegative"
            raise TranslationError(msg)
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.forward_scale, self.inverse_scale)
        ):
            msg = "spatial transform scalings must be finite and positive"
            raise TranslationError(msg)

    def _adopt_vectors(self) -> None:
        """Own the node and column vectors after checking their lengths."""
        size = math.prod(self.shape)
        for name in (
            "spatial_weights",
            "spectral_weights",
            "coordinates",
            "window",
            "absorber_rate",
            "on_axis",
        ):
            values = _frozen_array(
                getattr(self, name), np.complex128 if name == "on_axis" else np.float64
            )
            expected = (size, 2) if name == "coordinates" else (size,)
            if self.kind == "slab" and name in ("spectral_weights", "on_axis"):
                # The state's columns are (p, k_y), not the nodes: n_p*n_y of them.
                expected = (values.size,)
                if values.ndim != 1 or values.size % self.shape[1] != 0:
                    msg = (
                        f"slab {name} must be a vector of n_p*n_y column values, "
                        f"a multiple of n_y = {self.shape[1]}, got {values.shape}"
                    )
                    raise TranslationError(msg)
            if values.shape != expected:
                msg = f"spatial {name} must have shape {expected}, got {values.shape}"
                raise TranslationError(msg)
            object.__setattr__(self, name, values)

    @property
    def n_columns(self) -> int:
        """Return the number of state columns the geometry describes.

        Returns
        -------
        int
            ``n_r`` or ``n_x*n_y`` for the radial and Cartesian kinds, where columns
            and nodes coincide, and ``n_p*n_y`` for a slab, where they do not.
        """
        return int(self.spectral_weights.size)

    def _check_slab_columns(self) -> None:
        """Check a slab's per-column and per-row arrays agree with its two shapes."""
        if self.on_axis.size != self.spectral_weights.size:
            msg = "slab on_axis and spectral_weights must both have n_p*n_y entries"
            raise TranslationError(msg)
        rows = (self.shape[1], self.shape[0])
        # The plates carry no collar: the window and the rate are the y collar's,
        # the same at every x node of a row, so a backend may apply them per (p, y).
        for name in ("window", "absorber_rate"):
            values = getattr(self, name).reshape(rows)
            if np.any(values != values[:, :1]):
                msg = f"slab {name} must depend on y alone (constant along each x row)"
                raise TranslationError(msg)

    def _check_coordinates(self) -> None:
        """Validate the stated ordering and the transverse spectral meaning."""
        epsilon = np.finfo(float).eps
        if self.kind == "radial":
            if (
                np.any(self.axes[0] <= 0)
                or np.any(self.spectral_axes[0] <= 0)
                or np.any(np.diff(self.spectral_axes[0]) <= 0)
            ):
                msg = "radial spatial and spectral axes must be positive and increasing"
                raise TranslationError(msg)
            expected = np.column_stack((self.axes[0], np.zeros(self.shape[0])))
        else:
            pairs = (
                zip(self.axes[1:], self.spectral_axes, strict=True)
                if self.kind == "slab"
                else zip(self.axes, self.spectral_axes, strict=True)
            )
            for axis, spectral in pairs:
                if axis.size == 1:
                    # The line's y: one sample at the origin carrying only k_y = 0.
                    if axis[0] != 0 or spectral[0] != 0:
                        msg = "a Cartesian line's single y sample must be y = k_y = 0"
                        raise TranslationError(msg)
                    continue
                spacing = (axis[-1] - axis[0]) / (axis.size - 1)
                error = 32 * epsilon * max(float(np.max(np.abs(axis))), spacing)
                if np.any(np.abs(np.diff(axis) - spacing) > error):
                    msg = "Cartesian physical axes must be uniformly spaced"
                    raise TranslationError(msg)
                expected_frequency = 2 * np.pi * np.fft.fftfreq(axis.size, spacing)
                spectral_error = (
                    64 * epsilon * max(float(np.max(np.abs(expected_frequency))), 1.0)
                )
                if np.any(np.abs(spectral - expected_frequency) > spectral_error):
                    msg = (
                        "Cartesian spectral axes must be in FFT order "
                        "with reciprocal spacing"
                    )
                    raise TranslationError(msg)
            expected = np.column_stack(
                (
                    np.tile(self.axes[0], self.shape[1]),
                    np.repeat(self.axes[1], self.shape[0]),
                )
            )
        if not np.array_equal(self.coordinates, expected):
            msg = "spatial coordinates must match the axes with x fastest (ix + n_x*iy)"
            raise TranslationError(msg)
        count = (
            1
            if self.kind == "radial"
            else self.shape[1]
            if self.kind == "slab"
            else math.prod(self.shape)
        )
        if abs(count * self.forward_scale * self.inverse_scale - 1) > 32 * epsilon:
            msg = (
                "spatial scalings must normalize the QDHT or the unnormalised FFT pair"
            )
            raise TranslationError(msg)


@dataclass(frozen=True, eq=False)
class SeparableOperator:
    """Store a constant spatial operator and prefactor without coefficient volumes.

    Attributes
    ----------
    k_squared : numpy.ndarray
        Shared complex material wave-number square in m^-2, shape ``(n_omega,)``.
    kperp_sq : numpy.ndarray
        Transverse wave-number square in m^-2, shape ``(n_modes,)``, nonnegative.
    k_squared_low : numpy.ndarray
        Complex residual of the material square in m^-2, shape ``(n_omega,)``.
    kperp_sq_low : numpy.ndarray
        Real residual of the transverse square in m^-2, shape ``(n_modes,)``.
    frame : numpy.ndarray
        Shared real frame subtraction in m^-1, shape ``(n_omega,)``.
    spectral_rate : numpy.ndarray
        Shared spectral power absorption in m^-1, shape ``(n_omega,)``, nonnegative.
    prefactor_numerator : numpy.ndarray
        Shared purely imaginary numerator of the nonlinear prefactor, with the
        temporal crop scale included, shape ``(n_omega,)``.
    prefactor_clamped : numpy.ndarray
        Shared prefactor when the real longitudinal square is nonpositive, with the
        temporal crop scale included, shape ``(n_omega,)``.
    active : numpy.ndarray
        Physical frequency mask, bool, shape ``(n_omega,)``.
    evanescent_rate : float
        Largest evanescent field attenuation rate in m^-1, positive and finite.
    n_pol : int
        State polarisation components, 1 or 2; every coefficient is shared by them.
    z_span : tuple of float
        Valid propagation interval in m, finite and increasing.
    kappa_sq : numpy.ndarray or None
        Kernel schema 15, the hybrid planar kind only: each slab channel's complex
        transverse wave-number square ``kappa_p(omega)**2 = k_squared -
        beta_p(omega)**2`` in m^-2, shape ``(n_channels, n_omega)``, where
        ``beta_p = omega n_eff,p / c`` is the mode's own complex propagation
        constant (the wall term under the mode's loss policy). Column ``c`` belongs
        to channel ``c % n_channels`` (the channel index fastest). Zero at inactive
        frequencies. ``None`` (the default) for every other kind, which leaves the
        operator exactly as before.
    kperp_rate : numpy.ndarray or None
        Kernel schema 31: the transverse-wavevector power absorption rate
        ``alpha_perp`` of a :class:`solprop.boundaries.TransverseCollar`, in
        m^-1, one per column, shape ``(n_modes,)``, finite and nonnegative. It enters
        the operator on the active band only, beside the spectral rate,
        ``D -= alpha_perp / 2`` (SEP 22 section 5.3), and never the unitary phase.
        ``None`` (the default) is stored as zeros, whose operator is today's to the
        last bit. Refused nonzero on the hybrid planar kind.

    Raises
    ------
    TranslationError
        If arrays, scales, frequency counts or the valid interval disagree, or the
        transverse rate is negative, misshaped or set on the hybrid planar kind.
    """

    k_squared: Complex[np.ndarray, " n_k"]
    kperp_sq: Float[np.ndarray, " n_transverse"]
    k_squared_low: Complex[np.ndarray, " n_k_low"]
    kperp_sq_low: Float[np.ndarray, " n_transverse_low"]
    frame: Float[np.ndarray, " n_frame"]
    spectral_rate: Float[np.ndarray, " n_spectral_rate"]
    prefactor_numerator: Complex[np.ndarray, " n_numerator"]
    prefactor_clamped: Complex[np.ndarray, " n_clamped"]
    active: Bool[np.ndarray, " n_active"]
    evanescent_rate: float
    n_pol: int
    z_span: tuple[float, float]
    kappa_sq: Complex[np.ndarray, "n_channels n_kappa"] | None = None
    kperp_rate: Float[np.ndarray, " n_kperp_rate"] | None = None

    def __post_init__(self) -> None:
        """Own immutable coefficients and validate the separable representation."""
        for name in (
            "k_squared",
            "kperp_sq",
            "k_squared_low",
            "kperp_sq_low",
            "frame",
            "spectral_rate",
            "prefactor_numerator",
            "prefactor_clamped",
            "active",
        ):
            dtype = (
                np.bool_
                if name == "active"
                else np.complex128
                if name
                in (
                    "k_squared",
                    "k_squared_low",
                    "prefactor_numerator",
                    "prefactor_clamped",
                )
                else np.float64
            )
            values = _frozen_array(getattr(self, name), dtype)
            if values.ndim != 1 or values.size == 0:
                msg = f"separable {name} must be a nonempty vector"
                raise TranslationError(msg)
            if name == "kperp_sq_low" and values.size != self.kperp_sq.size:
                msg = "separable kperp_sq_low must match the transverse count"
                raise TranslationError(msg)
            if (
                name not in ("kperp_sq", "kperp_sq_low")
                and values.size != self.k_squared.size
            ):
                msg = f"separable {name} must match k_squared's frequency count"
                raise TranslationError(msg)
            object.__setattr__(self, name, values)
        if np.any(self.kperp_sq < 0) or np.any(self.spectral_rate < 0):
            msg = "separable kperp_sq and spectral_rate must be nonnegative"
            raise TranslationError(msg)
        if np.any(np.real(self.prefactor_numerator) != 0) or np.any(
            np.real(self.prefactor_clamped) != 0
        ):
            msg = "separable prefactors must be purely imaginary"
            raise TranslationError(msg)
        if self.n_pol not in (1, 2) or isinstance(self.n_pol, bool):
            msg = "separable n_pol must be 1 or 2"
            raise TranslationError(msg)
        if not math.isfinite(self.evanescent_rate) or self.evanescent_rate <= 0:
            msg = "separable evanescent_rate must be finite and positive"
            raise TranslationError(msg)
        if (
            len(self.z_span) != 2
            or not all(math.isfinite(value) for value in self.z_span)
            or not self.z_span[0] < self.z_span[1]
        ):
            msg = "separable z_span must be finite and increasing"
            raise TranslationError(msg)
        if self.kappa_sq is not None:
            kappa = _frozen_array(self.kappa_sq, np.complex128)
            if (
                kappa.ndim != 2
                or kappa.shape[0] == 0
                or kappa.shape[1] != self.k_squared.size
                or self.kperp_sq.size % kappa.shape[0] != 0
            ):
                msg = (
                    "separable kappa_sq must be (n_channels, n_omega), with n_omega "
                    "matching k_squared and n_channels dividing the column count"
                )
                raise TranslationError(msg)
            if np.any(kappa[:, ~self.active] != 0):
                msg = "separable kappa_sq must vanish at inactive frequencies"
                raise TranslationError(msg)
            object.__setattr__(self, "kappa_sq", kappa)
        self._adopt_kperp_rate()

    def _adopt_kperp_rate(self) -> None:
        """Own the transverse rate, zeros when absent, and validate it.

        Raises
        ------
        TranslationError
            If it is not one finite nonnegative rate per column, or is nonzero on the
            hybrid planar kind.
        """
        rate = (
            np.zeros(self.kperp_sq.size)
            if self.kperp_rate is None
            else _frozen_array(self.kperp_rate, np.float64)
        )
        rate.setflags(write=False)
        if rate.shape != self.kperp_sq.shape or np.any(rate < 0):
            msg = (
                "separable kperp_rate must be one nonnegative rate per transverse "
                f"column, shape {self.kperp_sq.shape}"
            )
            raise TranslationError(msg)
        if self.kappa_sq is not None and np.any(rate):
            msg = (
                "separable kperp_rate is refused on the hybrid planar kind: its "
                "k_y-only transverse collar is not part of SEP 22"
            )
            raise TranslationError(msg)
        object.__setattr__(self, "kperp_rate", rate)

    @property
    def n_channels(self) -> int:
        """Return the number of slab channels the columns cycle through.

        Returns
        -------
        int
            The rows of `kappa_sq`, or 1 when it is absent.
        """
        return 1 if self.kappa_sq is None else int(self.kappa_sq.shape[0])

    @property
    def value_shape(self) -> tuple[int, int, int, int]:
        """Return the full state shape represented by these shared coefficients.

        Returns
        -------
        tuple of int
            ``(1, n_modes, n_pol, n_omega)`` without allocating that volume.
        """
        return (1, self.kperp_sq.size, self.n_pol, self.k_squared.size)


@dataclass(frozen=True)
class RateBounds:
    """Bounds on a separable operator's real part, for the exponent budget.

    SEP 22 section 4.5: the decaying factor ``exp(lambda_minus h)`` of the inverse
    propagator and the growing one ``exp(lambda_plus h)`` of a gain medium overflow
    once their exponent passes ``log(DBL_MAX)``. Each bound is the largest over the
    frequencies of a sufficient per-frequency bound, and the terms are those of the
    frequency where the loss bound is largest.

    Attributes
    ----------
    loss : float
        ``lambda_minus``, a bound on ``max(-Re D)`` in m^-1, nonnegative.
    gain : float
        ``lambda_plus``, a bound on ``max(Re D)`` in m^-1, nonnegative.
    propagating : float
        The largest propagating attenuation at the binding frequency, m^-1.
    evanescent : float
        The capped evanescent attenuation there, ``min(|Im k_z|, evanescent_rate)``.
    spectral : float
        Half the spectral rate there.
    transverse : float
        Half the transverse collar's largest rate, applied on active frequencies.
    """

    loss: float
    gain: float
    propagating: float
    evanescent: float
    spectral: float
    transverse: float


def rate_bounds(coefficients: SeparableOperator) -> RateBounds:
    """Bound the real part of a separable operator without forming it.

    The search of :func:`solprop.spatial.attenuation_extremes` per slab channel (one
    on every kind but the hybrid planar), so the cost is
    ``O(n_omega log n_modes)``. On an active frequency the loss bound is the larger
    of the propagating attenuation's positive part and the capped evanescent one,
    plus half the largest transverse rate, plus half the spectral rate; on an
    inactive one it is half the spectral rate alone (SEP 22, Q6). The gain bound is
    the negated smallest propagating attenuation where it is negative.

    Parameters
    ----------
    coefficients : SeparableOperator
        The operator.

    Returns
    -------
    RateBounds
        The bounds and the binding frequency's terms.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.spatial import SeparableOperator, rate_bounds
    >>> operator = SeparableOperator(
    ...     k_squared=np.array([0j, 4 + 0j]), kperp_sq=np.array([0.0, 9.0]),
    ...     k_squared_low=np.zeros(2, complex), kperp_sq_low=np.zeros(2),
    ...     frame=np.zeros(2), spectral_rate=np.array([6.0, 0.0]),
    ...     prefactor_numerator=np.zeros(2, complex),
    ...     prefactor_clamped=np.zeros(2, complex),
    ...     active=np.array([False, True]), evanescent_rate=2.0, n_pol=1,
    ...     z_span=(0.0, 1.0), kperp_rate=np.array([0.0, 4.0]),
    ... )
    >>> bounds = rate_bounds(operator)
    >>> bounds.loss, bounds.evanescent, bounds.transverse
    (4.0, 2.0, 2.0)
    """
    data = coefficients
    active = data.active
    n_omega = data.k_squared.size
    largest = np.full(n_omega, -np.inf)
    smallest = np.full(n_omega, np.inf)
    evanescent = np.zeros(n_omega)
    columns = np.arange(data.kperp_sq.size)
    for channel in range(data.n_channels):
        mine = columns[columns % data.n_channels == channel]
        square = (
            data.k_squared
            if data.kappa_sq is None
            else data.k_squared - data.kappa_sq[channel]
        )
        high, low, ev = attenuation_extremes(
            square,
            data.kperp_sq[mine],
            square_low=data.k_squared_low,
            kperp_sq_low=data.kperp_sq_low[mine],
        )
        largest = np.maximum(largest, high)
        smallest = np.minimum(smallest, low)
        evanescent = np.maximum(evanescent, ev)
    propagating = np.where(active, np.maximum(largest, 0.0), 0.0)
    capped = np.where(active, np.minimum(evanescent, data.evanescent_rate), 0.0)
    assert data.kperp_rate is not None  # adopted as zeros
    transverse = 0.5 * float(np.max(data.kperp_rate, initial=0.0))
    loss = (
        np.maximum(propagating, capped)
        + np.where(active, transverse, 0.0)
        + 0.5 * data.spectral_rate
    )
    binding = int(np.argmax(loss))
    gain = np.where(active & np.isfinite(smallest), np.maximum(-smallest, 0.0), 0.0)
    return RateBounds(
        loss=float(loss[binding]),
        gain=float(np.max(gain, initial=0.0)),
        propagating=float(propagating[binding]),
        evanescent=float(capped[binding]),
        spectral=float(0.5 * data.spectral_rate[binding]),
        transverse=transverse if bool(active[binding]) else 0.0,
    )


def _frozen_array(values: np.ndarray, dtype: type) -> np.ndarray:
    """Own a finite, contiguous array without freezing a caller's writable buffer.

    Parameters
    ----------
    values : numpy.ndarray
        Numerical coefficients or geometry values to copy.
    dtype : type
        Required seam dtype: float64, complex128 or bool.

    Returns
    -------
    numpy.ndarray
        Finite, contiguous, read-only copy with the input shape.
    """
    result = np.array(values, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        msg = "spatial contract arrays must be finite"
        raise TranslationError(msg)
    result.setflags(write=False)
    return result


def separable_values(
    coefficients: SeparableOperator,
    columns: slice | None = None,
) -> tuple[
    Complex[np.ndarray, "1 n_modes n_pol n_omega"],
    Complex[np.ndarray, "n_modes n_pol n_omega"],
]:
    """Evaluate a separable operator and prefactor for the host reference.

    The reference may select bounded transverse columns. Compiled backends
    evaluate the same formula per element and retain only shared vectors.

    Parameters
    ----------
    coefficients : SeparableOperator
        Constant coefficients, already validated and in the internal convention.
    columns : slice or None, optional
        Transverse columns to evaluate; None selects all columns.

    Returns
    -------
    tuple of numpy.ndarray
        Linear rate in m^-1, shape ``(1, n_modes, n_pol, n_omega)``, and nonlinear
        prefactor, shape ``(n_modes, n_pol, n_omega)``.
    """
    root, propagating, attenuation = longitudinal_root(coefficients, columns)
    transverse_rate = _kperp_rate(coefficients, columns)[:, None]
    # The transverse rate joins the field attenuation inside the active factor (SEP
    # 22, Q6), so an inactive bin's operator is today's whatever the collar; a zero
    # rate leaves every element as it was, signed zeros included: -a - 0 = -a.
    operator = (
        (-attenuation - 0.5 * transverse_rate)
        - 1j * (np.real(root) - coefficients.frame)
    ) * coefficients.active - 0.5 * coefficients.spectral_rate
    prefactor = np.broadcast_to(coefficients.prefactor_clamped, root.shape).copy()
    np.divide(
        coefficients.prefactor_numerator,
        np.real(root),
        out=prefactor,
        where=propagating,
    )
    prefactor *= coefficients.active
    shape = (root.shape[0], coefficients.n_pol, len(coefficients.k_squared))
    return (
        np.broadcast_to(operator[:, None, :], shape)[None],
        np.broadcast_to(prefactor[:, None, :], shape),
    )


def _kperp_rate(
    coefficients: SeparableOperator, columns: slice | None
) -> Float[np.ndarray, " n_columns"]:
    """Return the transverse rate of the selected columns.

    Parameters
    ----------
    coefficients : SeparableOperator
        The operator.
    columns : slice or None
        Transverse columns; None selects all.

    Returns
    -------
    numpy.ndarray
        The power rate in m^-1, one per selected column.
    """
    rate = coefficients.kperp_rate
    assert rate is not None  # adopted as zeros in __post_init__
    return rate if columns is None else rate[columns]


def longitudinal_root(
    coefficients: SeparableOperator,
    columns: slice | None = None,
) -> tuple[
    Complex[np.ndarray, "n_modes n_omega"],
    Bool[np.ndarray, "n_modes n_omega"],
    Float[np.ndarray, "n_modes n_omega"],
]:
    """Return the longitudinal wave number, its propagating test and its field rate.

    The one evaluation of the root the stages use: the principal square root ``k_z``
    of the compensated square ``q = (k^2 - k_perp^2) + (k_low^2 - k_perp,low^2)``
    (with a slab channel's wall term subtracted last), the test ``Re q > 0`` that
    separates propagating from evanescent columns, and the field attenuation,
    ``Im k_z`` where propagating and ``min(|Im k_z|, evanescent_rate)`` where not.
    :func:`separable_values` builds the operator from it, and the far-field
    statistics read it (:func:`axial_factors`), so both see the same root
    (SEP 23 section 6).

    Parameters
    ----------
    coefficients : SeparableOperator
        Constant coefficients in the internal convention.
    columns : slice or None, optional
        Transverse columns to evaluate; None selects all columns.

    Returns
    -------
    tuple of numpy.ndarray
        ``k_z`` in m^-1 (complex), the propagating mask (bool) and the field
        attenuation in m^-1, each of shape ``(n_columns, n_omega)``. Inactive bins
        are not masked here: every consumer applies ``coefficients.active``.
    """
    transverse = (
        coefficients.kperp_sq if columns is None else coefficients.kperp_sq[columns]
    )
    transverse_low = (
        coefficients.kperp_sq_low
        if columns is None
        else coefficients.kperp_sq_low[columns]
    )
    square = (coefficients.k_squared[None, :] - transverse[:, None]) + (
        coefficients.k_squared_low[None, :] - transverse_low[:, None]
    )
    if coefficients.kappa_sq is not None:
        # q = k^2 eps_co - kappa_p^2 - k_y^2, the wall term subtracted last so a
        # free-space column's arithmetic is untouched (kernel schema 15).
        indices = np.arange(coefficients.kperp_sq.size)
        channels = (indices if columns is None else indices[columns]) % (
            coefficients.n_channels
        )
        square = square - coefficients.kappa_sq[channels]
    root = np.sqrt(square)
    propagating = np.real(square) > 0
    attenuation = np.where(
        propagating,
        np.imag(root),
        np.minimum(np.abs(np.imag(root)), coefficients.evanescent_rate),
    )
    return root, propagating, attenuation


def axial_admittance(
    coefficients: SeparableOperator,
    omega: Float[np.ndarray, " n_omega"],
    columns: slice,
) -> Complex[np.ndarray, "n_chunk n_omega"]:
    """Return the dimensionless forward admittance ``c k_z / omega`` of a chunk.

    The physical axial admittance is ``eps0 c`` times this value (SEP 34). It is the
    root of :func:`longitudinal_root` on active propagating bins at positive
    frequency, in the optics convention (positive imaginary part for loss), and zero
    elsewhere: a nonpropagating or inactive column is not a forward power port.

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

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.constants import C
    >>> from solprop.translate.spatial import SeparableOperator, axial_admittance
    >>> omega = np.array([0.0, C, 2 * C])
    >>> vacuum = SeparableOperator(
    ...     k_squared=(omega / C) ** 2 + 0j, kperp_sq=np.array([0.0, 2.25]),
    ...     k_squared_low=np.zeros(3, complex), kperp_sq_low=np.zeros(2),
    ...     frame=np.zeros(3), spectral_rate=np.zeros(3),
    ...     prefactor_numerator=np.zeros(3, complex),
    ...     prefactor_clamped=np.zeros(3, complex),
    ...     active=np.array([False, True, True]), evanescent_rate=200.0, n_pol=1,
    ...     z_span=(0.0, 1.0),
    ... )
    >>> np.round(axial_admittance(vacuum, omega, slice(None)).real, 12).tolist()
    [[0.0, 1.0, 1.0], [0.0, 0.0, 0.661437827766]]
    """
    root, propagating, _ = longitudinal_root(coefficients, columns)
    live = propagating & coefficients.active[None, :] & (omega > 0)[None, :]
    admittance = np.zeros(root.shape, dtype=np.complex128)
    np.divide(C * root, omega[None, :], out=admittance, where=live)
    return admittance


def axial_factors(
    coefficients: SeparableOperator,
    omega: Float[np.ndarray, " n_omega"],
    columns: slice,
) -> tuple[
    Complex[np.ndarray, "n_chunk n_omega"], Float[np.ndarray, "4 n_chunk n_omega"]
]:
    """Return a chunk's admittance and absorption rates for the far-field statistics.

    The weights of the far-field statistics (SEP 23 sections 4.8 and 4.9, measured
    by SEP 34's axial flux), from :func:`longitudinal_root`. The admittance is
    :func:`axial_admittance`. The four power rates, in the order of
    :data:`solprop.spatial.ABSORBED_CHANNELS`, are the operator's spectral rate on
    every bin, as the stages apply it; the transverse collar's rate on active bins
    (zero without one); ``2 min(|Im k_z|, cap)`` on active evanescent bins; and
    ``2 Im k_z`` on active propagating bins, the material loss (negative for gain).
    Together they are twice the negated real part of the operator; weighted by the
    real admittance, their power is the physical flux the linear flow removes. The
    evanescent channel, whose admittance is zero, removes no physical flux.

    Parameters
    ----------
    coefficients : SeparableOperator
        The run's constant coefficients.
    omega : numpy.ndarray
        Physical angular frequencies in rad/s, shape ``(n_omega,)``.
    columns : slice
        The transverse columns of the chunk.

    Returns
    -------
    tuple of numpy.ndarray
        The dimensionless complex admittance, shape ``(n_chunk, n_omega)``, and the
        rates in m^-1, shape ``(4, n_chunk, n_omega)``.
    """
    root, propagating, attenuation = longitudinal_root(coefficients, columns)
    active = coefficients.active[None, :]
    rates = np.zeros((4, *root.shape))
    rates[0] = coefficients.spectral_rate[None, :]
    # The transverse collar's rate acts on the active band, as the operator applies
    # it (SEP 22, Q6).
    rates[1] = np.where(active, _kperp_rate(coefficients, columns)[:, None], 0.0)
    rates[2] = np.where(active & ~propagating, 2.0 * attenuation, 0.0)
    rates[3] = np.where(active & propagating, 2.0 * attenuation, 0.0)
    return axial_admittance(coefficients, omega, columns), rates


@dataclass(frozen=True, eq=False)
class FluxPlan:
    """The plane-measure coefficients of a spatial run at any position (SEP 34).

    Kernel schema 35. The run's statistics read the axial admittance of the medium
    at the plane they measure through :meth:`at`: a uniform medium's is the run's
    own separable coefficients, and a radial gas path's replaces the material square
    by its z table, so the root is :func:`longitudinal_root`'s at that position.

    Attributes
    ----------
    base : SeparableOperator
        Transverse squares, wall terms and active band, with the material square at
        the entrance.
    material : solprop.translate.chebyshev.PiecewiseChebyshev or None
        The complex material square ``(omega n(omega, z) / c)**2`` in m^-2 along z,
        value shape ``(n_omega,)``, for a radial medium that varies; None for a
        uniform medium.
    admission : solprop.adiabatic.AdiabaticAdmission or None
        The varying path's adiabatic certificate (SEP 34 section 4.5); None for a
        uniform medium, which needs none.
    """

    base: SeparableOperator
    material: PiecewiseChebyshev | None = None
    admission: AdiabaticAdmission | None = None

    def at(self, z: float) -> SeparableOperator:
        """Return the coefficients of the plane at ``z``.

        Parameters
        ----------
        z : float
            Position in m inside the translated span.

        Returns
        -------
        SeparableOperator
            The coefficients whose root is the plane's ``k_z``.
        """
        if self.material is None:
            return self.base
        square = np.asarray(self.material(float(z)), np.complex128)
        return replace(self.base, k_squared=np.where(self.base.active, square, 0))


def _far_field_columns(
    geometry: "SpatialGeometry | FreeGrid | RadialGrid | None",  # noqa: UP037
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the Cartesian columns a far-field statistic reads, or refuse the geometry.

    Parameters
    ----------
    geometry : SpatialGeometry, solprop.grid.FreeGrid, solprop.grid.RadialGrid or None
        The run's geometry record, or the grid of a saved plane.

    Returns
    -------
    tuple of numpy.ndarray
        The stored axes ``k_x`` and ``k_y`` in rad/m, the spectral weights and the
        on-axis weights.

    Raises
    ------
    OutputError
        For a radial or planar-hybrid geometry, or none at all.
    """
    if isinstance(geometry, FreeGrid):
        return geometry.k_x, geometry.k_y, geometry.spectral_weights, geometry.on_axis
    if isinstance(geometry, SpatialGeometry) and geometry.kind == "free":
        k_x, k_y = geometry.spectral_axes
        return k_x, k_y, geometry.spectral_weights, geometry.on_axis
    kind = (
        "a radial grid"
        if isinstance(geometry, RadialGrid)
        or (isinstance(geometry, SpatialGeometry) and geometry.kind == "radial")
        else "the hybrid planar kind, whose columns are slab modes,"
        if isinstance(geometry, SpatialGeometry)
        else "a problem without spatial geometry"
    )
    msg = (
        f"the far-field statistics {FAR_FIELD_STATISTICS} need a Cartesian grid or a "
        f"line, and {kind} has no Cartesian transverse wavevector to map onto a mask "
        "plane"
    )
    raise OutputError(msg)


def aperture_tables(
    spec: StatsSpec,
    grids: Grids | TimeGrid,
    geometry: "SpatialGeometry | FreeGrid | RadialGrid | None",  # noqa: UP037
) -> ApertureTables | None:
    """Form the far-field statistics' tables once, for every backend (SEP 23 §6).

    The per-aperture constants -- the profile's half-width, ``D/2`` or, under
    quadrature weighting, the corrected ``D_q/2`` of
    :func:`solprop.spatial.quadrature_half_width` (unchanged on a line, whose edges
    are straight) -- and the scale table of :func:`solprop.spatial.aperture_scale`,
    with the geometry's column axes, weights and on-axis vector and the spec's
    regions. Every backend calls this at construction and
    :func:`solprop.stats.collect` calls it for a saved plane, so all of them use the
    same scale to the bit.

    Parameters
    ----------
    spec : solprop.stats.StatsSpec
        The request and its far-field tables.
    grids : solprop.translate.masks.Grids or solprop.grid.TimeGrid
        Anything carrying the physical frequency axis ``omega`` in rad/s.
    geometry : SpatialGeometry, solprop.grid.FreeGrid, solprop.grid.RadialGrid or None
        The run's geometry record or a saved plane's grid.

    Returns
    -------
    solprop.spatial.ApertureTables or None
        The tables, or ``None`` when the spec names no far-field statistic.

    Raises
    ------
    OutputError
        For a radial or planar-hybrid geometry or none at all; and on a line, for an
        aperture whose centre is off ``y = 0`` or a region whose ``kappa_y`` interval
        excludes zero, which would select nothing.
    """
    if not set(spec.names) & set(FAR_FIELD_STATISTICS):
        return None
    k_x, k_y, weights, on_axis = _far_field_columns(geometry)
    apertures = spec.apertures
    regions = spec.spectral_regions
    line = k_y.size == 1
    if line:
        if apertures is not None and np.any(np.asarray(apertures.centres)[:, 1] != 0):
            msg = (
                "on a line uniform in y an aperture is a slit along x, so its y centre "
                "must be zero"
            )
            raise OutputError(msg)
        excluded = np.flatnonzero(~((regions[:, 2] <= 0) & (regions[:, 3] > 0)))
        if excluded.size:
            msg = (
                "on a line every column has kappa_y = 0, so regions "
                f"{excluded.tolist()}, whose kappa_y interval excludes zero, would "
                "select nothing"
            )
            raise OutputError(msg)
    if apertures is None:
        empty = np.empty(0)
        return ApertureTables(
            k_x=k_x,
            k_y=k_y,
            weights=weights,
            on_axis=on_axis,
            regions=regions,
            centres=np.empty((0, 2)),
            half_widths=empty,
            edges=empty,
            distances=empty,
            scale=np.empty((0, grids.omega.size)),
        )
    diameters = np.asarray(apertures.diameters)
    edges = (
        np.zeros(diameters.size)
        if apertures.edges is None
        else np.asarray(apertures.edges)
    )
    half_widths = diameters / 2
    if apertures.weighting == "quadrature" and not line:
        half_widths = quadrature_half_width(half_widths, edges, apertures.profile)
    distances = np.asarray(apertures.distances)
    return ApertureTables(
        k_x=k_x,
        k_y=k_y,
        weights=weights,
        on_axis=on_axis,
        regions=regions,
        centres=np.asarray(apertures.centres),
        half_widths=half_widths,
        edges=edges,
        distances=distances,
        scale=aperture_scale(distances, grids.omega),
        profile=apertures.profile,
        exponent=1 if apertures.weighting == "quadrature" else 2,
    )

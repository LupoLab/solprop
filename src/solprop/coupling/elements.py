# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Linear optical elements acting on a spatial field between planes (SEP 31 §4.6).

Each element is a frozen record with an ``apply`` method that takes a
:class:`SpatialState` -- a radial or Cartesian spectrum with its temporal grid, its
homogeneous medium and its oriented local frame -- and returns the new state with an
:class:`ElementReport`. The stored spectrum keeps the layout and units of
:meth:`solprop.pulses.SpatialPulse.build`: the internal (FFT) temporal convention,
a raw order-zero Hankel spectrum in V m on a :class:`solprop.grid.RadialGrid` or a
raw two-dimensional DFT in V/m on a :class:`solprop.grid.FreeGrid`, shape
``(n_modes, n_pol, n_omega)``. Every physical expression in this module -- the
propagation phase :math:`e^{+ik_zL}`, a lens :math:`e^{-ikr^2/(2f)}`, a mask, a
Jones matrix or a reflection coefficient -- is written in the optics convention of
``docs/dev/conventions.md`` section 2 and crosses to the stored spectrum as its
complex conjugate, as :func:`solprop.optics.stored_factor` does for a transfer
function. Transverse transforms are not conjugated: they act on positions, which
are the same in both conventions.

A :class:`~solprop.coupling.plane.FieldPlane` is carried through a sequence of these
elements by :func:`solprop.coupling.transport`, which converts it to this state and
back and updates its reference, geometry, medium and metric; the state and the
:class:`Element` protocol are the interface a new element is written against. Every
refusal raises :class:`solprop.exceptions.CouplingError`.

Energies are physical: the axial Poynting flux of the medium at the plane,
:class:`solprop.spatial.SpatialFlux` and :func:`solprop.spatial.spatial_energy`
(SEP 34). Every report closes its own ledger in joules,

.. math:: U_{\rm out} = U_{\rm in} - L_{\rm specified} - U_{\rm removed}
          + \epsilon_{\rm approximation},

so the approximation error is the signed part of the energy change that neither a
declared loss nor an intentional removal explains.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Complex, Float
from numpy.typing import ArrayLike
from scipy.special import j0

from solprop import fourier
from solprop.constants import C
from solprop.exceptions import CouplingError
from solprop.grid import FreeGrid, LineGrid, RadialGrid, TimeGrid
from solprop.optics import (
    Chain,
    ChirpedMirror,
    GratingPair,
    Material,
    PrismPair,
    TaylorPhase,
    TransferFunction,
    stored_factor,
)
from solprop.spatial import SpatialFlux, SpatialMedium, aperture_profile, spatial_energy

__all__ = [
    "Alignment",
    "Aperture",
    "Element",
    "ElementReport",
    "ElementResult",
    "FreeSpace",
    "Mirror",
    "PlanarInterface",
    "PolarisationTransform",
    "ScreenCertificate",
    "SpatialState",
    "SpectralCalibration",
    "SpectralTransfer",
    "ThinLens",
    "TransmissionMask",
]

_EPS = float(np.finfo(float).eps)
_CHUNK = 2**20
"""Complex elements of a direct-summation kernel block (16 MiB)."""
_OCCUPIED = 1e-12
"""Relative content below which a node or column counts as empty in a diagnostic."""
_FRAME_TOLERANCE = 1e-12
"""Orthonormality tolerance of a local frame and of a declared direction."""

_Positions = Float[np.ndarray, " n_nodes"]
_Band = Float[np.ndarray, " n_band"]
Transmission = Callable[[_Positions, _Positions, _Band], ArrayLike]
"""A field transmission ``T(x, y, omega)`` in the optics convention.

Positions in m, shape ``(n_nodes,)``; positive in-band angular frequencies in rad/s,
shape ``(n_band,)``. The result broadcasts to ``(n_nodes, n_band)``.
"""

SpectralFunction = Callable[[Float[np.ndarray, " n_band"]], ArrayLike]
"""A function of positive angular frequency in rad/s, shape ``(n_band,)``."""


# --------------------------------------------------------------------- state record


@dataclass(frozen=True, eq=False)
class SpatialState:
    """A spatial field at one plane, with its grids, medium and local frame.

    The plain input of every element; :func:`solprop.coupling.transport` builds it
    from a :class:`~solprop.coupling.plane.FieldPlane` (the internal conjugate of
    the plane's samples, its grid's frame axes as rows). The single component of a scalar state
    (``n_pol = 1``) is polarised along the frame's x axis; two components are the
    optics-convention ``(x, y)`` Jones components stored conjugated, as
    :meth:`solprop.pulses.SpatialPulse.build` stores them.

    Attributes
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid of the spectrum.
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation. A :class:`solprop.grid.LineGrid` (per unit
        width) and a planar grid (slab amplitudes) are refused.
    spectrum : numpy.ndarray
        Internal-convention Hankel/temporal spectrum in V m or Cartesian
        DFT/temporal spectrum in V/m, complex128, shape ``(n_modes, n_pol,
        n_omega)``; exactly zero outside the active band and at nonpositive
        frequency. Stored as a read-only copy.
    medium : solprop.spatial.SpatialMedium
        Homogeneous medium at the plane: every parameter path is constant.
    frame : numpy.ndarray
        Right-handed orthonormal local frame in laboratory coordinates, shape
        ``(3, 3)``: the rows are the transverse x and y axes and the forward
        propagation direction. Default: the identity. Stored as a read-only copy.

    Raises
    ------
    CouplingError
        If the representation, the spectrum's shape, finiteness or band, or the
        frame is invalid, or the medium varies along z.
    """

    grid: TimeGrid
    space: RadialGrid | FreeGrid
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"]
    medium: SpatialMedium
    frame: Float[np.ndarray, "3 3"] = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        """Validate and freeze the representation, the spectrum and the frame."""
        space = self.space
        if isinstance(space, LineGrid) or not isinstance(space, RadialGrid | FreeGrid):
            msg = (
                "transport takes a radial or Cartesian field plane; a line grid "
                "(per unit width) and a planar grid (slab amplitudes) are not field "
                f"planes of a finite beam, got {type(space).__name__}"
            )
            raise CouplingError(msg)
        spectrum = np.array(self.spectrum, dtype=np.complex128, copy=True)
        if (
            spectrum.ndim != 3
            or spectrum.shape[0] != space.n_columns
            or spectrum.shape[1] not in (1, 2)
            or spectrum.shape[2] != self.grid.n_omega
            or not np.all(np.isfinite(spectrum))
        ):
            msg = (
                "a spatial state's spectrum must be finite with shape "
                f"({space.n_columns}, 1 or 2, {self.grid.n_omega}), got "
                f"{spectrum.shape}"
            )
            raise CouplingError(msg)
        if np.any(spectrum[:, :, ~_band(self.grid)]):
            msg = (
                "a spatial state is exactly zero outside the active band and at "
                "nonpositive frequency, as a launch or a saved plane is"
            )
            raise CouplingError(msg)
        if not all(path.is_constant for path in self.medium.parameter_paths.values()):
            msg = (
                "transport between planes needs a homogeneous medium; a medium that "
                "varies along z is an ordinary propagation, not a free-space element"
            )
            raise CouplingError(msg)
        frame = _unit_frame(self.frame)
        spectrum.setflags(write=False)
        object.__setattr__(self, "spectrum", spectrum)
        object.__setattr__(self, "frame", frame)

    @property
    def n_pol(self) -> int:
        """Return the number of stored polarisation components.

        Returns
        -------
        int
            One for a scalar state, two for ``(x, y)``.
        """
        return int(self.spectrum.shape[1])

    def with_spectrum(
        self,
        spectrum: Complex[np.ndarray, "n_columns n_components n_omega"],
        *,
        space: RadialGrid | FreeGrid | None = None,
        medium: SpatialMedium | None = None,
        frame: Float[np.ndarray, "3 3"] | None = None,
    ) -> "SpatialState":  # noqa: UP037 - beartype evaluates the class body
        """Return a state with a new spectrum and, optionally, new plane data.

        Parameters
        ----------
        spectrum : numpy.ndarray
            New internal spectrum, shape ``(n_columns, n_components, n_omega)``.
        space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or None, optional
            New transverse representation; None keeps this one.
        medium : solprop.spatial.SpatialMedium or None, optional
            New medium; None keeps this one.
        frame : numpy.ndarray or None, optional
            New local frame, shape ``(3, 3)``; None keeps this one.

        Returns
        -------
        SpatialState
            The validated new state.
        """
        return SpatialState(
            self.grid,
            self.space if space is None else space,
            spectrum,
            self.medium if medium is None else medium,
            self.frame if frame is None else frame,
        )


def _unit_frame(frame: ArrayLike) -> Float[np.ndarray, "3 3"]:
    """Return a validated, read-only right-handed orthonormal frame.

    Parameters
    ----------
    frame : array_like
        Rows x, y and forward direction in laboratory coordinates, shape ``(3, 3)``.

    Returns
    -------
    numpy.ndarray
        A read-only float64 copy.

    Raises
    ------
    CouplingError
        If the frame is not finite, orthonormal and right-handed.
    """
    values = np.array(frame, dtype=np.float64, copy=True)
    if (
        values.shape != (3, 3)
        or not np.all(np.isfinite(values))
        or np.max(np.abs(values @ values.T - np.eye(3))) > _FRAME_TOLERANCE
        or np.linalg.det(values) <= 0
    ):
        msg = (
            "a local frame is a right-handed orthonormal 3x3 array whose rows are "
            "the x, y and forward directions"
        )
        raise CouplingError(msg)
    values.setflags(write=False)
    return values


# --------------------------------------------------------------------- reports


@dataclass(frozen=True, eq=False)
class ScreenCertificate:
    r"""The single-field thin-screen admission record of SEP 31 §4.6.

    A thin screen is exact in the local reference measure
    :math:`U_0=\int Y_0|\tilde E|^2\,d\kappa\,d\omega/2` with
    :math:`Y_0=\operatorname{Re}n\,\varepsilon_0c`; the physical axial energy
    :math:`U_Y` weighs each plane wave by :math:`\cos\theta` in a transparent
    medium. With :math:`1-\cos\theta_*=\epsilon_m/8`, the defect
    :math:`d(E)=(1-\cos\theta_*)U_{0,\rm cone}+U_{0,\rm tail}` bounds
    :math:`U_0-U_Y`, and the screen is admitted when
    :math:`[d(E_{\rm in})+d(E_{\rm out})]/U_Y(E_{\rm in})` plus the numerical
    error is at most :math:`\epsilon_m`.

    Attributes
    ----------
    model_error : float
        The required budget :math:`\epsilon_m`, in (0, 1).
    cone_cosine : float
        :math:`\cos\theta_* = 1 - \epsilon_m/8`.
    defect_in, defect_out : float
        :math:`d(E)` of the input and output fields, J (reference measure).
    reference_energy : float
        :math:`U_Y(E_{\rm in})`, J.
    numerical_error : float
        Relative numerical and domain error added to the ratio: the transform
        round trip's energy discrepancy bound :math:`2\delta+\delta^2` plus the
        field's share at the window's edge on input and output.
    ratio : float
        The admitted quantity, at most `model_error`; zero for a zero input.
    exact_zero : bool
        True when the input is exactly zero, the exact separate case.
    scope : str
        Always ``"single field"``: the certificate applies only to this field,
        not to a reusable map over a subspace.
    """

    model_error: float
    cone_cosine: float
    defect_in: float
    defect_out: float
    reference_energy: float
    numerical_error: float
    ratio: float
    exact_zero: bool
    scope: str = "single field"


@dataclass(frozen=True)
class SpectralCalibration:
    """Which affine spectral phase a frequency-only element removes or keeps.

    A plane's reference record (SEP 31 §4.5) accumulates the carrier phase and the
    delay of its path; an element that removes them from its own response leaves
    them unknown to that record unless the removal is recorded.

    Attributes
    ----------
    removes_phase : bool or None
        True if the element drops its constant phase at the reference frequency,
        False if it keeps it, None if unknown.
    removes_delay : bool or None
        True if the element drops its group delay at the reference frequency,
        False if it keeps it, None if unknown.
    reference_wavelength : float or None
        Vacuum wavelength in m at which the removal is referenced, or None.
    note : str
        How the record was obtained.

    Examples
    --------
    >>> from solprop.coupling.elements import SpectralCalibration
    >>> SpectralCalibration(True, True, 800e-9, "measured compressor").removes_delay
    True
    """

    removes_phase: bool | None
    removes_delay: bool | None
    reference_wavelength: float | None
    note: str = ""


@dataclass(frozen=True, eq=False)
class ElementReport:
    """The accounting of one element's application (SEP 31 §4.7).

    Energies are the physical axial flux of the plane's medium (SEP 34) on the
    whole represented plane. They satisfy ``energy_out = energy_in -
    specified_loss - removed + approximation_error`` exactly.

    Attributes
    ----------
    element : str
        A short description of the element.
    position : int
        Index of the element in a transport sequence; 0 when applied alone.
    energy_in, energy_out : float
        Energy crossing the input and output planes, J.
    specified_loss : float
        Loss the element declares: material absorption, reflection, a mask's
        local loss :math:`L_0` or a transfer function's attenuation, J.
    removed : float
        Energy intentionally removed: an explicit angular band limit, or the
        bins a transfer function sets to exactly zero, J.
    approximation_error : float
        Signed remainder of the ledger, J: the thin-screen axial-flux defect,
        obliquity at an interface, output-window truncation and resampling, or
        rounding for an exact element. Never absorption.
    numerical_error : float
        Estimated numerical energy error of the representation, J.
    spectrum_in, spectrum_out : numpy.ndarray
        Energy per temporal bin at the input and output planes, J, shape
        ``(n_omega,)``; each sums to its energy.
    certificate : ScreenCertificate or None
        The thin-screen certificate of a lens, mask, aperture or tilt.
    calibration : tuple of SpectralCalibration
        One record per frequency-only factor; empty for a spatial element.
    reconstruction : bool
        True for an operation that is not a passive physical element: a negative
        length, an inverse phase or a backward propagation.
    diagnostics : collections.abc.Mapping
        Element-specific numbers (sampling and wrap shares, bounds), read-only.
    notes : tuple of str
        Element-specific statements, such as the factors of a chain.
    """

    element: str
    position: int
    energy_in: float
    energy_out: float
    specified_loss: float
    removed: float
    approximation_error: float
    numerical_error: float
    spectrum_in: Float[np.ndarray, " n_omega_in"]
    spectrum_out: Float[np.ndarray, " n_omega_out"]
    certificate: ScreenCertificate | None = None
    calibration: tuple[SpectralCalibration, ...] = ()
    reconstruction: bool = False
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the arrays and the diagnostics."""
        for name in ("spectrum_in", "spectrum_out"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        frozen = MappingProxyType(dict(self.diagnostics))
        object.__setattr__(self, "diagnostics", frozen)


@dataclass(frozen=True, eq=False)
class ElementResult:
    """The state after an element and the report of its application.

    Attributes
    ----------
    state : SpatialState
        The field at the element's output plane.
    report : ElementReport
        Its accounting.
    """

    state: SpatialState
    report: ElementReport


@runtime_checkable
class Element(Protocol):
    """A linear element between two planes."""

    def apply(self, state: SpatialState) -> ElementResult:
        """Apply the element to a field plane.

        Parameters
        ----------
        state : SpatialState
            The field at the element's input plane.

        Returns
        -------
        ElementResult
            The output plane and the report.
        """
        ...


# --------------------------------------------------------------------- shared numerics


def _band(grid: TimeGrid) -> Bool[np.ndarray, " n_omega"]:
    """Return the bins that can carry field: active and at positive frequency.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shape ``(n_omega,)``.
    """
    return grid.active_mask & (grid.omega > 0)


def _index(medium: SpatialMedium, grid: TimeGrid) -> Complex[np.ndarray, " n_omega"]:
    """Return the medium's complex index on the band, zero elsewhere.

    Parameters
    ----------
    medium : solprop.spatial.SpatialMedium
        Homogeneous medium.
    grid : solprop.grid.TimeGrid
        Temporal grid.

    Returns
    -------
    numpy.ndarray
        Refractive index, dimensionless, shape ``(n_omega,)``.
    """
    band = _band(grid)
    index = np.zeros(grid.n_omega, dtype=np.complex128)
    index[band] = medium.index(grid.omega[band])
    return index


def _same_medium(first: SpatialMedium, second: SpatialMedium, grid: TimeGrid) -> bool:
    """Return whether two media have the same index on the band.

    Parameters
    ----------
    first, second : solprop.spatial.SpatialMedium
        The media.
    grid : solprop.grid.TimeGrid
        Temporal grid.

    Returns
    -------
    bool
        True for the same object or exactly equal indices on every band bin.
    """
    return first is second or bool(
        np.array_equal(_index(first, grid), _index(second, grid))
    )


def _transverse_squares(space: RadialGrid | FreeGrid) -> Float[np.ndarray, " n_modes"]:
    """Return the squared transverse wave number of each stored column.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation.

    Returns
    -------
    numpy.ndarray
        ``k_r**2`` or ``kperp_sq`` in m^-2, shape ``(n_modes,)``.
    """
    return space.k_r**2 if isinstance(space, RadialGrid) else space.kperp_sq


def _to_field(
    space: RadialGrid | FreeGrid, spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"]
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Synthesise the field samples from a stored transverse spectrum.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation.
    spectrum : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    numpy.ndarray
        Field samples in the internal temporal convention, V/m times the temporal
        DFT scaling, at the grid's nodes (x fastest on a Cartesian grid).
    """
    if isinstance(space, RadialGrid):
        return np.moveaxis(space.hankel.inverse(np.moveaxis(spectrum, 0, -1)), -1, 0)
    shape = (space.n_y, space.n_x, *spectrum.shape[1:])
    return np.fft.ifft2(spectrum.reshape(shape), axes=(0, 1)).reshape(spectrum.shape)


def _to_spectrum(
    space: RadialGrid | FreeGrid, samples: Complex[np.ndarray, "n_modes n_pol n_omega"]
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Transform field samples to the stored transverse spectrum.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation.
    samples : numpy.ndarray
        Field samples at the grid's nodes, shape ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    numpy.ndarray
        The raw Hankel or DFT spectrum of the samples, the same shape.
    """
    if isinstance(space, RadialGrid):
        return np.moveaxis(space.hankel.forward(np.moveaxis(samples, 0, -1)), -1, 0)
    shape = (space.n_y, space.n_x, *samples.shape[1:])
    return np.fft.fft2(samples.reshape(shape), axes=(0, 1)).reshape(samples.shape)


def _nodes(space: RadialGrid | FreeGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return the x and y coordinates of the grid's nodes.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation; radial nodes lie on the x axis.

    Returns
    -------
    tuple of numpy.ndarray
        ``x`` and ``y`` in m, each shape ``(n_nodes,)``, x fastest.
    """
    if isinstance(space, RadialGrid):
        return np.asarray(space.r), np.zeros(space.n_r)
    return (
        np.tile(space.x, space.n_y),
        np.repeat(space.y, space.n_x),
    )


def _flux(state: SpatialState) -> SpatialFlux:
    """Resolve the axial-flux measure of the state's plane.

    Parameters
    ----------
    state : SpatialState
        The plane; its medium is homogeneous, so the position is immaterial.

    Returns
    -------
    solprop.spatial.SpatialFlux
        The measure of SEP 34.
    """
    return SpatialFlux.resolve(state.grid, state.space, state.medium, z=0.0)


def _energy_spectrum(
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"], flux: SpatialFlux
) -> Float[np.ndarray, " n_omega"]:
    """Split :func:`solprop.spatial.spatial_energy` into temporal bins.

    The same weighted sum, accumulated per bin rather than in total: the axial
    admittance of :meth:`solprop.spatial.SpatialFlux.admittance`, the transverse
    Parseval weights and the analytic-signal endpoint weights of
    :func:`solprop.fourier.energy_weights`.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``.
    flux : solprop.spatial.SpatialFlux
        The plane's measure.

    Returns
    -------
    numpy.ndarray
        Energy per bin in J, shape ``(n_omega,)``.
    """
    grid = flux.grid
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    total = np.zeros(grid.n_omega)
    for columns in flux.chunks():
        real = np.real(flux.admittance(columns))
        density = np.sum(np.abs(spectrum[columns]) ** 2, axis=1)
        total += np.sum(flux.space.spectral_weights[columns, None] * real * density, 0)
    return flux.energy_scale * weights * total


@dataclass(frozen=True)
class _Measured:
    """The physical energy of a plane, in total and per bin.

    Attributes
    ----------
    energy : float
        Total energy in J (:func:`solprop.spatial.spatial_energy`).
    spectrum : numpy.ndarray
        Energy per bin in J.
    """

    energy: float
    spectrum: Float[np.ndarray, " n_omega"]


def _measure(state: SpatialState, flux: SpatialFlux) -> _Measured:
    """Measure the physical energy of a state.

    Parameters
    ----------
    state : SpatialState
        The plane.
    flux : solprop.spatial.SpatialFlux
        Its measure.

    Returns
    -------
    _Measured
        Total and per-bin energy, J.
    """
    return _Measured(
        spatial_energy(state.spectrum, flux), _energy_spectrum(state.spectrum, flux)
    )


@dataclass(frozen=True)
class _Angular:
    """The reference-measure content of a plane split by propagation angle.

    Attributes
    ----------
    cone : float
        Content of propagating columns inside the cone, J.
    tail : float
        Content outside the cone, nonpropagating columns included, J.
    evanescent : float
        Content of nonpropagating columns, J; part of `tail`.
    """

    cone: float
    tail: float
    evanescent: float

    @property
    def total(self) -> float:
        """Return the whole reference-measure content.

        Returns
        -------
        float
            ``cone + tail`` in J.
        """
        return self.cone + self.tail


def _angular_content(
    state: SpatialState, flux: SpatialFlux, cone_cosine: float
) -> _Angular:
    r"""Split the local reference measure :math:`U_0` by propagation angle.

    :math:`U_0` weighs each column with :math:`Y_0=\operatorname{Re}n\,\varepsilon_0c`
    in place of the axial admittance; the obliquity of a propagating column is the
    admittance over the index, :math:`\cos\theta`.

    Parameters
    ----------
    state : SpatialState
        The plane.
    flux : solprop.spatial.SpatialFlux
        Its measure.
    cone_cosine : float
        :math:`\cos\theta_*`; columns with a smaller obliquity are the tail.

    Returns
    -------
    _Angular
        The split content, J.
    """
    grid = state.grid
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    index = np.real(_index(state.medium, grid))
    cone = tail = evanescent = 0.0
    for columns in flux.chunks():
        density = np.sum(np.abs(state.spectrum[columns]) ** 2, axis=1)
        content = (
            flux.energy_scale
            * state.space.spectral_weights[columns, None]
            * (weights * index)
            * density
        )
        admittance = np.real(flux.admittance(columns))
        cosine = np.divide(
            admittance,
            index,
            out=np.zeros_like(admittance),
            where=index > 0,
        )
        propagating = admittance > 0
        inside = propagating & (cosine >= cone_cosine)
        cone += float(np.sum(content[inside]))
        tail += float(np.sum(content[~inside]))
        evanescent += float(np.sum(content[~propagating]))
    return _Angular(cone, tail, evanescent)


def _reference_norm(
    space: RadialGrid | FreeGrid,
    grid: TimeGrid,
    index: Float[np.ndarray, " n_omega"],
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> float:
    """Return the squared reference-measure norm of a spectrum, up to a constant.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation.
    grid : solprop.grid.TimeGrid
        Temporal grid.
    index : numpy.ndarray
        Real index per bin, zero off band, shape ``(n_omega,)``.
    spectrum : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    float
        ``sum v_j gamma_k n_k |X_jpk|^2``, in the units of ``|X|^2`` m^-2 s.
    """
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    density = np.sum(np.abs(spectrum) ** 2, axis=1)
    return float(np.sum(space.spectral_weights[:, None] * (weights * index) * density))


def _edge_nodes(space: RadialGrid | FreeGrid) -> Bool[np.ndarray, " n_nodes"]:
    """Return the nodes of the window's outer band.

    The declared collar where there is one, otherwise the outer tenth of each
    half-width.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation.

    Returns
    -------
    numpy.ndarray
        Boolean mask over the nodes, shape ``(n_nodes,)``.
    """
    if space.window_factor > 0:
        return np.asarray(space.window < 1)
    if isinstance(space, RadialGrid):
        return np.asarray(space.r > 0.9 * space.radius)
    x, y = _nodes(space)
    return np.asarray(
        (np.abs(x) > 0.9 * np.max(np.abs(space.x)))
        | (np.abs(y) > 0.9 * np.max(np.abs(space.y)))
    )


def _node_content(
    space: RadialGrid | FreeGrid,
    grid: TimeGrid,
    index: Float[np.ndarray, " n_omega"],
    samples: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> Float[np.ndarray, " n_modes"]:
    """Return each node's reference-measure content, up to a constant.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation.
    grid : solprop.grid.TimeGrid
        Temporal grid.
    index : numpy.ndarray
        Real index per bin, zero off band.
    samples : numpy.ndarray
        Field samples at the nodes, shape ``(n_modes, n_pol, n_omega)``.

    Returns
    -------
    numpy.ndarray
        Area-weighted content per node, shape ``(n_modes,)``.
    """
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    density = np.sum(np.abs(samples) ** 2, axis=1)
    radial = isinstance(space, RadialGrid)
    area = space.radial_weights if radial else space.spatial_weights
    return area * np.sum((weights * index) * density, axis=1)


def _edge_share(state: SpatialState) -> float:
    """Return the share of the field in the window's outer band.

    Parameters
    ----------
    state : SpatialState
        The plane.

    Returns
    -------
    float
        Reference-measure share of the field at the edge nodes, in [0, 1]; zero for
        a zero field.
    """
    index = np.real(_index(state.medium, state.grid))
    content = _node_content(
        state.space, state.grid, index, _to_field(state.space, state.spectrum)
    )
    total = float(np.sum(content))
    if total == 0:
        return 0.0
    return float(np.sum(content[_edge_nodes(state.space)])) / total


def _round_trip_error(state: SpatialState) -> float:
    r"""Return the relative energy error bound of one synthesis round trip.

    The raw QDHT is only approximately self-inverse; a field that is synthesised
    and transformed back differs by :math:`\delta` in the reference norm, which
    bounds the relative energy discrepancy by :math:`2\delta+\delta^2` (SEP 31
    §4.7). On a Cartesian grid it is rounding.

    Parameters
    ----------
    state : SpatialState
        The plane.

    Returns
    -------
    float
        :math:`2\delta+\delta^2`; zero for a zero field.
    """
    index = np.real(_index(state.medium, state.grid))
    norm = _reference_norm(state.space, state.grid, index, state.spectrum)
    if norm == 0:
        return 0.0
    back = _to_spectrum(state.space, _to_field(state.space, state.spectrum))
    delta = math.sqrt(
        _reference_norm(state.space, state.grid, index, back - state.spectrum) / norm
    )
    return 2 * delta + delta * delta


@dataclass(frozen=True)
class _Extras:
    """The optional parts of an element report.

    Attributes
    ----------
    certificate : ScreenCertificate or None
        Thin-screen certificate.
    calibration : tuple of SpectralCalibration
        Affine-phase records.
    reconstruction : bool
        Whether the operation is a reconstruction.
    diagnostics : collections.abc.Mapping
        Element-specific numbers.
    notes : tuple of str
        Element-specific statements.
    """

    certificate: ScreenCertificate | None = None
    calibration: tuple[SpectralCalibration, ...] = ()
    reconstruction: bool = False
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def _report(  # noqa: PLR0913 - the ledger's terms, each named at the call
    element: str,
    measured_in: _Measured,
    measured_out: _Measured,
    *,
    specified_loss: float = 0.0,
    removed: float = 0.0,
    numerical_error: float = 0.0,
    extras: _Extras | None = None,
) -> ElementReport:
    """Close an element's ledger and build its report.

    Parameters
    ----------
    element : str
        Description of the element.
    measured_in, measured_out : _Measured
        Energies of the input and output planes.
    specified_loss : float, optional
        Declared loss, J.
    removed : float, optional
        Intentional removal, J.
    numerical_error : float, optional
        Estimated numerical error, J.
    extras : _Extras or None, optional
        The optional report fields.

    Returns
    -------
    ElementReport
        The report, with the approximation error that closes the ledger.
    """
    extras = _Extras() if extras is None else extras
    approximation = measured_out.energy - measured_in.energy + specified_loss + removed
    return ElementReport(
        element=element,
        position=0,
        energy_in=measured_in.energy,
        energy_out=measured_out.energy,
        specified_loss=specified_loss,
        removed=removed,
        approximation_error=approximation,
        numerical_error=numerical_error,
        spectrum_in=measured_in.spectrum,
        spectrum_out=measured_out.spectrum,
        certificate=extras.certificate,
        calibration=extras.calibration,
        reconstruction=extras.reconstruction,
        diagnostics=extras.diagnostics,
        notes=extras.notes,
    )


def _spectral_values(
    values: float | complex | SpectralFunction,
    omega: Float[np.ndarray, " n_band"],
    name: str,
) -> Complex[np.ndarray, " n_band"]:
    """Evaluate a constant or a function of frequency on the band.

    Parameters
    ----------
    values : float, complex or callable
        A constant, or a function of positive angular frequency.
    omega : numpy.ndarray
        Band frequencies in rad/s, shape ``(n_band,)``.
    name : str
        The quantity, for messages.

    Returns
    -------
    numpy.ndarray
        Complex values, shape ``(n_band,)``.

    Raises
    ------
    CouplingError
        If the values are not finite or do not match the band.
    """
    result = values(omega) if callable(values) else values
    array = np.broadcast_to(np.asarray(result, dtype=np.complex128), omega.shape)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite on every band frequency"
        raise CouplingError(msg)
    return np.array(array)


def _check_model_error(model_error: float) -> None:
    """Refuse a model-error budget outside (0, 1).

    Parameters
    ----------
    model_error : float
        The budget.

    Raises
    ------
    CouplingError
        If it is not finite and strictly between zero and one.
    """
    if not (math.isfinite(model_error) and 0 < model_error < 1):
        msg = f"model_error is a required budget in (0, 1), got {model_error!r}"
        raise CouplingError(msg)


# --------------------------------------------------------------------- thin screens


def _occupied(content: Float[np.ndarray, " n_modes"]) -> Bool[np.ndarray, " n_modes"]:
    """Return the nodes whose content is above the diagnostic floor.

    Parameters
    ----------
    content : numpy.ndarray
        Content per node.

    Returns
    -------
    numpy.ndarray
        Boolean mask; all False for a zero field.
    """
    peak = float(np.max(content)) if content.size else 0.0
    return content > _OCCUPIED * peak if peak > 0 else np.zeros(content.shape, bool)


def _phase_step(
    space: RadialGrid | FreeGrid,
    transmission: Complex[np.ndarray, "n_modes n_band"],
    occupied: Bool[np.ndarray, " n_modes"],
) -> float:
    """Return a screen's largest phase change between occupied neighbouring nodes.

    A step of pi or more samples the screen's phase below its Nyquist rate: the
    spatial frequencies it generates alias onto the grid.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation.
    transmission : numpy.ndarray
        Optics-convention transmission at the nodes, shape ``(n_modes, n_band)``.
    occupied : numpy.ndarray
        Nodes that hold field, shape ``(n_modes,)``.

    Returns
    -------
    float
        The largest step in rad over neighbours that are both occupied and both
        transmitting; zero if there are none.
    """
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if isinstance(space, RadialGrid):
        pairs.append(
            (transmission[1:], transmission[:-1], occupied[1:] & occupied[:-1])
        )
    else:
        values = transmission.reshape(space.n_y, space.n_x, -1)
        mask = occupied.reshape(space.n_y, space.n_x)
        pairs.append((values[:, 1:], values[:, :-1], mask[:, 1:] & mask[:, :-1]))
        pairs.append((values[1:], values[:-1], mask[1:] & mask[:-1]))
    step = 0.0
    for later, earlier, both in pairs:
        ratio = later * np.conj(earlier)
        valid = both[..., None] & (np.abs(ratio) > 0)
        if np.any(valid):
            step = max(step, float(np.max(np.abs(np.angle(ratio[valid])))))
    return step


def _apply_screen(  # noqa: PLR0913 - the screen and its declarations
    state: SpatialState,
    transmission: Complex[np.ndarray, "n_modes n_band"],
    *,
    element: str,
    model_error: float,
    phase_only: bool,
    gradient: Float[np.ndarray, "n_modes n_band"] | None = None,
    notes: tuple[str, ...] = (),
) -> ElementResult:
    """Apply a thin local screen and certify it against its budget.

    Parameters
    ----------
    state : SpatialState
        The input plane.
    transmission : numpy.ndarray
        Optics-convention field transmission at the nodes on the band bins,
        shape ``(n_modes, n_band)``.
    element : str
        Description, for the report.
    model_error : float
        The required budget in (0, 1).
    phase_only : bool
        Whether the screen is declared unit-modulus; its local loss is then zero.
    gradient : numpy.ndarray or None, optional
        The screen's analytic phase gradient magnitude in rad/m at the nodes,
        shape ``(n_modes, n_band)``, when it is known. Sampled phases wrap, so
        only an analytic gradient detects a step beyond pi.
    notes : tuple of str, optional
        Statements for the report.

    Returns
    -------
    ElementResult
        The output plane and its report with the certificate.

    Raises
    ------
    CouplingError
        If the screen amplifies anywhere, its phase is undersampled where the
        field is, or the certificate exceeds the budget.
    """
    _check_model_error(model_error)
    if not np.all(np.isfinite(transmission)):
        msg = f"{element}: the transmission must be finite"
        raise CouplingError(msg)
    if np.max(np.abs(transmission), initial=0.0) > 1 + 8 * _EPS:
        msg = (
            f"{element}: a passive screen has |T| <= 1; a gain screen is not "
            "a linear passive element"
        )
        raise CouplingError(msg)
    band = _band(state.grid)
    index = np.real(_index(state.medium, state.grid))
    samples = _to_field(state.space, state.spectrum)
    occupied = _occupied(_node_content(state.space, state.grid, index, samples))
    step = _phase_step(state.space, transmission, occupied)
    if gradient is not None and np.any(occupied):
        analytic = float(np.max(gradient[occupied])) * _spacing(state.space)
        step = max(step, analytic)
    if step >= math.pi:
        msg = (
            f"{element}: the screen's phase changes by {step:.3g} rad between "
            "neighbouring nodes where the field is, beyond the grid's Nyquist rate; "
            "refine the transverse grid"
        )
        raise CouplingError(msg)
    factor = np.ones((state.space.n_columns, state.grid.n_omega), dtype=np.complex128)
    factor[:, band] = fourier.to_internal(transmission)
    after = state.with_spectrum(_to_spectrum(state.space, factor[:, None, :] * samples))
    flux = _flux(state)
    measured_in = _measure(state, flux)
    measured_out = _measure(after, flux)
    # Numerical and domain error: the synthesis round trip, plus the field at the
    # window's edge before and after, which the representation may wrap or clip.
    numerical = _round_trip_error(state) + _edge_share(state) + _edge_share(after)
    certificate = _screen_certificate(
        state, after, flux, model_error=model_error, numerical=numerical
    )
    local_loss = 0.0
    if not phase_only:
        cosine = certificate.cone_cosine
        local_loss = (
            _angular_content(state, flux, cosine).total
            - _angular_content(after, flux, cosine).total
        )
    report = _report(
        element,
        measured_in,
        measured_out,
        specified_loss=local_loss,
        numerical_error=numerical * measured_in.energy,
        extras=_Extras(
            certificate=certificate, diagnostics={"phase_step": step}, notes=notes
        ),
    )
    return ElementResult(after, report)


def _screen_certificate(
    before: SpatialState,
    after: SpatialState,
    flux: SpatialFlux,
    *,
    model_error: float,
    numerical: float,
) -> ScreenCertificate:
    """Evaluate the thin-screen defect on input and output and admit or refuse.

    Parameters
    ----------
    before, after : SpatialState
        The input and output planes, on one representation and medium.
    flux : solprop.spatial.SpatialFlux
        Their common measure.
    model_error : float
        The budget in (0, 1).
    numerical : float
        Relative numerical error added to the ratio.

    Returns
    -------
    ScreenCertificate
        The admitted certificate.

    Raises
    ------
    CouplingError
        If a nonzero input carries no forward flux or the defect ratio exceeds
        the budget.
    """
    cosine = 1 - model_error / 8
    if not np.any(before.spectrum):
        return ScreenCertificate(model_error, cosine, 0.0, 0.0, 0.0, 0.0, 0.0, True)
    content_in = _angular_content(before, flux, cosine)
    content_out = _angular_content(after, flux, cosine)
    defect_in = (1 - cosine) * content_in.cone + content_in.tail
    defect_out = (1 - cosine) * content_out.cone + content_out.tail
    reference = spatial_energy(before.spectrum, flux)
    if reference <= 0:
        msg = "a thin screen's input carries no forward flux to certify against"
        raise CouplingError(msg)
    ratio = (defect_in + defect_out) / reference + numerical
    if ratio > model_error:
        msg = (
            f"the thin-screen defect is {ratio:.3e} of the input energy, beyond "
            f"model_error={model_error:g}: the field leaves the paraxial cone "
            "cos(theta*) = 1 - model_error/8 or the transform is not resolved; "
            "relax the budget only if a larger axial-flux error is acceptable"
        )
        raise CouplingError(msg)
    return ScreenCertificate(
        model_error, cosine, defect_in, defect_out, reference, numerical, ratio, False
    )


def _evaluate_transmission(
    transmission: Transmission,
    x: Float[np.ndarray, " n_nodes"],
    y: Float[np.ndarray, " n_nodes"],
    omega: Float[np.ndarray, " n_band"],
    element: str,
) -> Complex[np.ndarray, "n_nodes n_band"]:
    """Evaluate a transmission callable and check its shape.

    Parameters
    ----------
    transmission : callable
        ``T(x, y, omega)`` in the optics convention.
    x, y : numpy.ndarray
        Node positions in m, shape ``(n_nodes,)``.
    omega : numpy.ndarray
        Band frequencies in rad/s, shape ``(n_band,)``.
    element : str
        Description, for messages.

    Returns
    -------
    numpy.ndarray
        The transmission, shape ``(n_nodes, n_band)``.

    Raises
    ------
    CouplingError
        If the result does not broadcast to ``(n_nodes, n_band)``.
    """
    values = np.asarray(transmission(x, y, omega), dtype=np.complex128)
    try:
        return np.array(np.broadcast_to(values, (x.size, omega.size)))
    except ValueError as error:
        msg = (
            f"{element}: the transmission must broadcast to (n_nodes, n_band) = "
            f"({x.size}, {omega.size}), got {values.shape}"
        )
        raise CouplingError(msg) from error


def _symmetric_transmission(
    transmission: Transmission,
    space: RadialGrid,
    omega: Float[np.ndarray, " n_band"],
    element: str,
) -> Complex[np.ndarray, "n_nodes n_band"]:
    """Evaluate a transmission on a radial grid, refusing an asymmetric one.

    The order-zero Hankel representation holds a cylindrically symmetric field; a
    screen that depends on the azimuth would break it silently.

    Parameters
    ----------
    transmission : callable
        ``T(x, y, omega)``.
    space : solprop.grid.RadialGrid
        Radial representation.
    omega : numpy.ndarray
        Band frequencies in rad/s.
    element : str
        Description, for messages.

    Returns
    -------
    numpy.ndarray
        The transmission at the radial nodes, shape ``(n_r, n_band)``.

    Raises
    ------
    CouplingError
        If the transmission depends on the azimuth.
    """
    radius = np.asarray(space.r)
    reference = _evaluate_transmission(
        transmission, radius, np.zeros_like(radius), omega, element
    )
    for angle in (0.7, 1.9, 3.3, 4.6):
        rotated = _evaluate_transmission(
            transmission,
            radius * math.cos(angle),
            radius * math.sin(angle),
            omega,
            element,
        )
        if np.max(np.abs(rotated - reference)) > 64 * _EPS * max(
            1.0, float(np.max(np.abs(reference)))
        ):
            msg = (
                f"{element} depends on the azimuth, and a radial grid holds only "
                "cylindrically symmetric fields: use a Cartesian FreeGrid"
            )
            raise CouplingError(msg)
    return reference


@dataclass(frozen=True, eq=False)
class ThinLens:
    r"""A thin paraxial lens, the phase screen :math:`e^{-ik(\omega)r^2/(2f(\omega))}`.

    :math:`k=\operatorname{Re}n\,\omega/c` is the wave number of the plane's
    medium and :math:`r` the distance from the plane's axis. The screen is
    certified per field against `model_error` (:class:`ScreenCertificate`).

    Attributes
    ----------
    focal_length : float or callable
        Focal length in m, finite and nonzero (negative diverges), or a function of
        positive angular frequency in rad/s returning it per band bin.
    model_error : float
        Required thin-screen budget in (0, 1).

    Examples
    --------
    >>> from solprop.coupling.elements import ThinLens
    >>> ThinLens(0.5, model_error=1e-3).focal_length
    0.5
    """

    focal_length: float | SpectralFunction
    model_error: float

    def apply(self, state: SpatialState) -> ElementResult:
        """Apply the lens screen.

        Parameters
        ----------
        state : SpatialState
            The input plane.

        Returns
        -------
        ElementResult
            The output plane and its certified report.

        Raises
        ------
        CouplingError
            If a focal length is zero or not finite.
        """
        band = _band(state.grid)
        omega = state.grid.omega[band]
        focal = np.real(_spectral_values(self.focal_length, omega, "focal_length"))
        if np.any(focal == 0):
            msg = "a thin lens needs a nonzero focal length on every band bin"
            raise CouplingError(msg)
        wavenumber = np.real(_index(state.medium, state.grid))[band] * omega / C
        x, y = _nodes(state.space)
        square = x * x + y * y
        curvature = wavenumber / (2 * focal)
        transmission = np.exp(-1j * square[:, None] * curvature[None, :])
        gradient = np.sqrt(square)[:, None] * np.abs(2 * curvature)[None, :]
        label = (
            f"ThinLens(f={self.focal_length:g} m)"
            if isinstance(self.focal_length, float | int)
            else "ThinLens(chromatic f)"
        )
        return _apply_screen(
            state,
            transmission,
            element=label,
            model_error=self.model_error,
            phase_only=True,
            gradient=gradient,
        )


@dataclass(frozen=True, eq=False)
class Aperture:
    """An aperture built on the SEP 23 edge profile, with its weighting declared.

    :func:`solprop.spatial.aperture_profile` gives the edge profile ``W``. Its
    meaning is fixed by `weighting`, exactly as for the in-run aperture statistics
    of :class:`solprop.stats.ApertureSet`: ``'amplitude'`` (the default) takes
    ``W`` as the **field** transmission, so the power goes as ``W**2``;
    ``'quadrature'`` takes ``W`` as a **power** weight, applied to the field as
    ``sqrt(W)``. SEP 31 §4.6's intensity-defined aperture is the ``'quadrature'``
    case. A hard aperture (``W`` zero or one) is the same either way.

    Attributes
    ----------
    radius : float
        Half-width of the profile in m, finite and positive.
    model_error : float
        Required thin-screen budget in (0, 1).
    edge : float
        Edge width in m for ``tanh`` and ``rcos``, order for ``supergauss``,
        unused for ``hard``. Default 0.
    kind : {'hard', 'tanh', 'rcos', 'supergauss'}
        Edge profile. Default ``'hard'``.
    centre : tuple of float
        Centre ``(x, y)`` in m in the local frame; a radial grid needs the axis.
    weighting : {'amplitude', 'quadrature'}
        ``'amplitude'`` applies ``W`` to the field, ``'quadrature'`` applies
        ``sqrt(W)``. Default ``'amplitude'``.

    Examples
    --------
    >>> from solprop.coupling import Aperture
    >>> Aperture(1e-3, model_error=1e-2, weighting="quadrature").weighting
    'quadrature'
    """

    radius: float
    model_error: float
    edge: float = 0.0
    kind: Literal["hard", "tanh", "rcos", "supergauss"] = "hard"
    centre: tuple[float, float] = (0.0, 0.0)
    weighting: Literal["amplitude", "quadrature"] = "amplitude"

    def apply(self, state: SpatialState) -> ElementResult:
        """Apply the aperture.

        Parameters
        ----------
        state : SpatialState
            The input plane.

        Returns
        -------
        ElementResult
            The output plane and its certified report.

        Raises
        ------
        CouplingError
            If the weighting is unknown or an off-axis aperture is asked of a
            radial grid.
        """
        if self.weighting not in ("amplitude", "quadrature"):
            msg = f"weighting is 'amplitude' or 'quadrature', got {self.weighting!r}"
            raise CouplingError(msg)
        if isinstance(state.space, RadialGrid) and any(self.centre):
            msg = (
                "an off-axis aperture breaks cylindrical symmetry: use a Cartesian "
                "FreeGrid"
            )
            raise CouplingError(msg)
        x, y = _nodes(state.space)
        distance = np.hypot(x - self.centre[0], y - self.centre[1])
        weight = aperture_profile(distance, self.radius, self.edge, self.kind)
        amplitude = np.sqrt(weight) if self.weighting == "quadrature" else weight
        n_band = int(np.sum(_band(state.grid)))
        transmission = np.repeat(amplitude[:, None], n_band, axis=1).astype(complex)
        return _apply_screen(
            state,
            transmission,
            element=(
                f"Aperture(radius={self.radius:g} m, {self.kind}, {self.weighting})"
            ),
            model_error=self.model_error,
            phase_only=False,
            notes=(
                "power weight W applied to the field as sqrt(W)"
                if self.weighting == "quadrature"
                else "field transmission W; power goes as W**2",
            ),
        )


@dataclass(frozen=True, eq=False)
class TransmissionMask:
    """A complex field transmission, as a function or as samples at the nodes.

    Attributes
    ----------
    transmission : callable or array_like
        ``T(x, y, omega)`` in the optics convention (positions in m, shape
        ``(n_nodes,)``; band angular frequencies in rad/s, shape ``(n_band,)``;
        a result broadcasting to ``(n_nodes, n_band)``), or samples at the
        state's nodes, shape ``(n_nodes,)`` or ``(n_nodes, n_band)``. A
        function on a radial grid must not depend on the azimuth.
        ``|T| <= 1`` everywhere.
    model_error : float
        Required thin-screen budget in (0, 1).
    phase_only : bool
        Declare ``|T| = 1``: the local loss is then exactly zero. Default False.
    """

    transmission: Transmission | ArrayLike
    model_error: float
    phase_only: bool = False

    def apply(self, state: SpatialState) -> ElementResult:
        """Apply the mask.

        Parameters
        ----------
        state : SpatialState
            The input plane.

        Returns
        -------
        ElementResult
            The output plane and its certified report.

        Raises
        ------
        CouplingError
            If sampled values have the wrong shape, a declared phase-only mask
            changes the modulus, or a function depends on the azimuth on a
            radial grid.
        """
        band = _band(state.grid)
        omega = state.grid.omega[band]
        n_nodes = state.space.n_columns
        if callable(self.transmission):
            if isinstance(state.space, RadialGrid):
                values = _symmetric_transmission(
                    self.transmission, state.space, omega, "TransmissionMask"
                )
            else:
                x, y = _nodes(state.space)
                values = _evaluate_transmission(
                    self.transmission, x, y, omega, "TransmissionMask"
                )
        else:
            array = np.asarray(self.transmission, dtype=np.complex128)
            if array.ndim == 1:
                array = array[:, None]
            if (
                array.ndim != 2
                or array.shape[0] != n_nodes
                or array.shape[1]
                not in (
                    1,
                    omega.size,
                )
            ):
                msg = (
                    "TransmissionMask samples have shape (n_nodes,) or (n_nodes, "
                    f"n_band) = ({n_nodes}, {omega.size}), got {array.shape}"
                )
                raise CouplingError(msg)
            values = np.array(np.broadcast_to(array, (n_nodes, omega.size)))
        if self.phase_only and np.max(np.abs(np.abs(values) - 1), initial=0.0) > (
            8 * _EPS
        ):
            msg = "a mask declared phase-only must have unit modulus"
            raise CouplingError(msg)
        return _apply_screen(
            state,
            values,
            element="TransmissionMask",
            model_error=self.model_error,
            phase_only=self.phase_only,
        )


@dataclass(frozen=True, eq=False)
class Alignment:
    """A lateral displacement and a paraxial tilt of a Cartesian field.

    The field is tilted about the plane's origin and then displaced rigidly:
    ``E'(x, y) = T(x - d_x, y - d_y) E(x - d_x, y - d_y)`` with the optics-convention
    tilt ``T = exp(+i k (x sin(theta_x) + y sin(theta_y)))``, ``k`` the medium's
    real wave number. A positive displacement moves the beam towards +x (+y); a
    positive tilt makes it propagate towards +x (+y). The displacement is exact in
    the periodic DFT representation; the tilt is a phase screen certified against
    `model_error`. A radial grid refuses both.

    Attributes
    ----------
    displacement : tuple of float
        ``(d_x, d_y)`` in m, finite. Default no displacement.
    tilt : tuple of float
        ``(theta_x, theta_y)`` in rad, finite, each below pi/2 in magnitude.
        Default no tilt.
    model_error : float or None
        Thin-screen budget in (0, 1), required for a nonzero tilt.
    """

    displacement: tuple[float, float] = (0.0, 0.0)
    tilt: tuple[float, float] = (0.0, 0.0)
    model_error: float | None = None

    def apply(self, state: SpatialState) -> ElementResult:
        """Apply the tilt, then the displacement.

        Parameters
        ----------
        state : SpatialState
            The input plane.

        Returns
        -------
        ElementResult
            The output plane and its report.

        Raises
        ------
        CouplingError
            If a nonzero alignment is asked of a radial grid, or a tilt has no
            budget or exceeds it.
        """
        values = (*self.displacement, *self.tilt)
        if not all(math.isfinite(value) for value in values) or any(
            abs(angle) >= math.pi / 2 for angle in self.tilt
        ):
            msg = "alignment needs finite displacements and tilts below pi/2"
            raise CouplingError(msg)
        if isinstance(state.space, RadialGrid):
            if any(values):
                msg = (
                    "displacement and tilt break cylindrical symmetry: use a "
                    "Cartesian FreeGrid"
                )
                raise CouplingError(msg)
            flux = _flux(state)
            measured = _measure(state, flux)
            return ElementResult(state, _report("Alignment(none)", measured, measured))
        space = state.space
        result: ElementResult | None = None
        current = state
        if any(self.tilt):
            if self.model_error is None:
                msg = "a tilt is a phase screen and needs its model_error budget"
                raise CouplingError(msg)
            band = _band(state.grid)
            index = np.real(_index(state.medium, state.grid))[band]
            wavenumber = index * state.grid.omega[band] / C
            x, y = _nodes(space)
            projection = x * math.sin(self.tilt[0]) + y * math.sin(self.tilt[1])
            slope = math.hypot(math.sin(self.tilt[0]), math.sin(self.tilt[1]))
            result = _apply_screen(
                state,
                np.exp(1j * projection[:, None] * wavenumber[None, :]),
                element=f"Alignment(tilt={self.tilt} rad)",
                model_error=self.model_error,
                phase_only=True,
                gradient=np.broadcast_to(slope * wavenumber, (x.size, wavenumber.size)),
            )
            current = result.state
        shift = np.exp(
            -1j
            * (
                np.tile(space.k_x, space.n_y) * self.displacement[0]
                + np.repeat(space.k_y, space.n_x) * self.displacement[1]
            )
        )
        moved = current.with_spectrum(shift[:, None, None] * current.spectrum)
        flux = _flux(state)
        measured_in = _measure(state, flux)
        measured_out = _measure(moved, flux)
        label = f"Alignment(displacement={self.displacement} m, tilt={self.tilt} rad)"
        if result is None:
            return ElementResult(moved, _report(label, measured_in, measured_out))
        report = replace(
            result.report,
            element=label,
            energy_out=measured_out.energy,
            spectrum_out=measured_out.spectrum,
            approximation_error=measured_out.energy
            - measured_in.energy
            + result.report.specified_loss,
        )
        return ElementResult(moved, report)


# --------------------------------------------------------------------- uniform maps


def _uniform_result(
    before: SpatialState,
    after: SpatialState,
    *,
    element: str,
    removed_bins: Bool[np.ndarray, " n_omega"] | None = None,
    reflected: Float[np.ndarray, " n_omega"] | None = None,
    extras: _Extras | None = None,
) -> ElementResult:
    """Account for a spatially uniform map.

    A uniform map commutes with the transverse transform: on one medium every
    column keeps its obliquity, the loss per bin is the energy change per bin and
    the ledger closes to rounding. Across a medium change the loss is the declared
    reflection and the rest is obliquity.

    Parameters
    ----------
    before, after : SpatialState
        The input and output planes.
    element : str
        Description.
    removed_bins : numpy.ndarray or None, optional
        Bins set to exactly zero by the element; their input energy is reported as
        removed.
    reflected : numpy.ndarray or None, optional
        Declared reflectance per bin across a medium change; the specified loss is
        then ``sum(R * spectrum_in)``.
    extras : _Extras or None, optional
        The optional report fields.

    Returns
    -------
    ElementResult
        The output plane and its report.
    """
    measured_in = _measure(before, _flux(before))
    measured_out = _measure(after, _flux(after))
    removed = 0.0
    if removed_bins is not None:
        removed = float(np.sum(measured_in.spectrum[removed_bins]))
    if reflected is not None:
        loss = float(np.sum(reflected * measured_in.spectrum))
    else:
        kept = np.ones(before.grid.n_omega, bool)
        if removed_bins is not None:
            kept = ~removed_bins
        loss = float(np.sum((measured_in.spectrum - measured_out.spectrum)[kept]))
    report = _report(
        element,
        measured_in,
        measured_out,
        specified_loss=loss,
        removed=removed,
        extras=extras,
    )
    return ElementResult(after, report)


def _apply_jones(
    state: SpatialState, jones: Complex[np.ndarray, "n_band 2 2"]
) -> Complex[np.ndarray, "n_modes n_components n_omega"]:
    """Apply an optics-convention Jones matrix per band bin to the stored spectrum.

    A scalar state is polarised along x. It stays scalar when the map creates no
    y component and becomes a two-component state otherwise.

    Parameters
    ----------
    state : SpatialState
        The input plane.
    jones : numpy.ndarray
        Jones matrices acting on ``(E_x, E_y)``, shape ``(n_band, 2, 2)``.

    Returns
    -------
    numpy.ndarray
        The new internal spectrum, shape ``(n_modes, 1 or 2, n_omega)``.
    """
    band = _band(state.grid)
    stored = np.zeros((state.grid.n_omega, 2, 2), dtype=np.complex128)
    stored[:] = np.eye(2)
    stored[band] = fourier.to_internal(jones)
    spectrum = state.spectrum
    if state.n_pol == 1:
        if not np.any(stored[:, 1, 0]):
            return stored[None, None, :, 0, 0] * spectrum
        spectrum = np.concatenate([spectrum, np.zeros_like(spectrum)], axis=1)
    return np.einsum("kab,jbk->jak", stored, spectrum)


def _check_passive_jones(
    jones: Complex[np.ndarray, "n_band 2 2"], element: str
) -> None:
    """Refuse a Jones map that amplifies some polarisation.

    Parameters
    ----------
    jones : numpy.ndarray
        Jones matrices, shape ``(n_band, 2, 2)``.
    element : str
        Description, for messages.

    Raises
    ------
    CouplingError
        If a matrix is not finite or its largest singular value exceeds one.
    """
    if not np.all(np.isfinite(jones)):
        msg = f"{element}: the Jones matrices must be finite"
        raise CouplingError(msg)
    if jones.size and float(np.max(np.linalg.svd(jones, compute_uv=False))) > (
        1 + 8 * _EPS
    ):
        msg = f"{element}: a passive Jones map has singular values of at most one"
        raise CouplingError(msg)


@dataclass(frozen=True, eq=False)
class PolarisationTransform:
    """A spatially uniform Jones matrix in the plane's local ``(x, y)`` axes.

    Attributes
    ----------
    jones : array_like or callable
        Optics-convention 2x2 matrix acting on ``(E_x, E_y)``, or a function of
        positive angular frequency in rad/s returning shape ``(n_band, 2, 2)``.
        Passive: singular values of at most one.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.coupling.elements import PolarisationTransform
    >>> PolarisationTransform(np.diag([1.0, -1.0])).jones.shape
    (2, 2)
    """

    jones: ArrayLike | Callable[[Float[np.ndarray, " n_band"]], ArrayLike]

    def apply(self, state: SpatialState) -> ElementResult:
        """Apply the Jones matrix.

        Parameters
        ----------
        state : SpatialState
            The input plane.

        Returns
        -------
        ElementResult
            The output plane and its report.

        Raises
        ------
        CouplingError
            If the matrices have the wrong shape or amplify.
        """
        omega = state.grid.omega[_band(state.grid)]
        values = self.jones(omega) if callable(self.jones) else self.jones
        array = np.asarray(values, dtype=np.complex128)
        try:
            jones = np.array(np.broadcast_to(array, (omega.size, 2, 2)))
        except ValueError as error:
            msg = f"a Jones matrix is 2x2 per band bin, got shape {array.shape}"
            raise CouplingError(msg) from error
        _check_passive_jones(jones, "PolarisationTransform")
        after = state.with_spectrum(_apply_jones(state, jones))
        return _uniform_result(state, after, element="PolarisationTransform")


def _unit_vector(
    vector: Sequence[float] | np.ndarray, name: str
) -> Float[np.ndarray, " 3"]:
    """Return a normalised three-vector.

    Parameters
    ----------
    vector : sequence of float or numpy.ndarray
        Three finite components, not all zero.
    name : str
        The quantity, for messages.

    Returns
    -------
    numpy.ndarray
        The unit vector, shape ``(3,)``.

    Raises
    ------
    CouplingError
        If the vector is not a finite nonzero three-vector.
    """
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or not np.any(array):
        msg = f"{name} must be a finite nonzero three-vector"
        raise CouplingError(msg)
    return array / np.linalg.norm(array)


@dataclass(frozen=True, eq=False)
class Mirror:
    r"""A flat mirror: spectral p/s reflection and the incoming/outgoing frame map.

    Each beam has the right-handed local frame :math:`(\hat p,\hat s,\hat k)` with
    :math:`\hat p=\hat s\times\hat k`, for incoming and outgoing directions alike.
    A field with components :math:`(E_p,E_s)` leaves as
    :math:`(r_pE_p,r_sE_s)` on :math:`(\hat p_{\rm out},\hat s)`; there is no
    conjugation. The outgoing local axes have the same p/s components as the
    incoming ones, so the frame stays right-handed, and the improper reflection
    appears as a mirror image of the transverse pattern along p (SEP 31 §4.6).
    In this convention a perfect conductor at any incidence has
    :math:`r_s=-1, r_p=+1`, and a dielectric at normal incidence
    :math:`r_p=-r_s`; coefficients calibrated in another p/s convention must be
    converted first.

    Attributes
    ----------
    reflectivity_p, reflectivity_s : complex or callable
        Optics-convention amplitude reflection coefficients, constants or functions
        of positive angular frequency in rad/s; ``|r| <= 1``.
    incident : sequence of float or numpy.ndarray
        Incoming propagation direction in laboratory coordinates; must match the
        plane's forward axis.
    outgoing : sequence of float or numpy.ndarray
        Outgoing propagation direction; not parallel to `incident`.
    s_axis : sequence of float or numpy.ndarray
        Direction perpendicular to both, hence normal to the plane of incidence
        when there is one; at normal incidence it resolves the degeneracy. Its
        sign does not change the map.
    """

    reflectivity_p: complex | SpectralFunction
    reflectivity_s: complex | SpectralFunction
    incident: Sequence[float] | np.ndarray
    outgoing: Sequence[float] | np.ndarray
    s_axis: Sequence[float] | np.ndarray

    def __post_init__(self) -> None:
        """Check the geometry at construction."""
        self._geometry()

    def _geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate the directions and return them normalised.

        Returns
        -------
        tuple of numpy.ndarray
            Incoming direction, outgoing direction and s axis, each shape ``(3,)``.

        Raises
        ------
        CouplingError
            If the directions do not define a reflection with this s axis.
        """
        incident = _unit_vector(self.incident, "incident")
        outgoing = _unit_vector(self.outgoing, "outgoing")
        s_axis = _unit_vector(self.s_axis, "s_axis")
        if float(incident @ outgoing) > 1 - _FRAME_TOLERANCE:
            msg = "a mirror changes the propagation direction: outgoing == incident"
            raise CouplingError(msg)
        if (
            abs(float(s_axis @ incident)) > _FRAME_TOLERANCE
            or abs(float(s_axis @ outgoing)) > _FRAME_TOLERANCE
        ):
            msg = "the s axis must be perpendicular to both propagation directions"
            raise CouplingError(msg)
        return incident, outgoing, s_axis

    def apply(self, state: SpatialState) -> ElementResult:
        """Reflect the plane.

        Parameters
        ----------
        state : SpatialState
            The input plane, propagating along `incident`.

        Returns
        -------
        ElementResult
            The reflected plane in its outgoing frame, and the report.

        Raises
        ------
        CouplingError
            If the geometry is invalid, the plane does not propagate along
            `incident`, a coefficient exceeds one in modulus, or the mirror image
            cannot be represented exactly on a Cartesian grid.
        """
        incident, outgoing, s_axis = self._geometry()
        frame = state.frame
        if float(frame[2] @ incident) < 1 - _FRAME_TOLERANCE:
            msg = "the plane does not propagate along the mirror's incident direction"
            raise CouplingError(msg)
        p_in = np.cross(s_axis, incident)
        p_out = np.cross(s_axis, outgoing)
        rotation = np.array(
            [[p_in @ frame[0], p_in @ frame[1]], [s_axis @ frame[0], s_axis @ frame[1]]]
        )
        omega = state.grid.omega[_band(state.grid)]
        r_p = _spectral_values(self.reflectivity_p, omega, "reflectivity_p")
        r_s = _spectral_values(self.reflectivity_s, omega, "reflectivity_s")
        diagonal = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        diagonal[:, 0, 0] = r_p
        diagonal[:, 1, 1] = r_s
        jones = np.einsum("ba,kbc,cd->kad", rotation, diagonal, rotation)
        _check_passive_jones(jones, "Mirror")
        spectrum = _apply_jones(state, jones)
        if isinstance(state.space, FreeGrid):
            image = rotation.T @ np.diag([-1.0, 1.0]) @ rotation
            spectrum = _mirror_image(state.space, spectrum, image)
        new_frame = np.array(
            [
                rotation[0, 0] * p_out + rotation[1, 0] * s_axis,
                rotation[0, 1] * p_out + rotation[1, 1] * s_axis,
                outgoing,
            ]
        )
        after = state.with_spectrum(spectrum, frame=new_frame)
        return _uniform_result(
            state,
            after,
            element="Mirror",
            extras=_Extras(notes=("p = s x k for both beams; no conjugation",)),
        )


def _mirror_image(
    space: FreeGrid,
    spectrum: Complex[np.ndarray, "n_modes n_components n_omega"],
    image: Float[np.ndarray, "2 2"],
) -> Complex[np.ndarray, "n_modes n_components n_omega"]:
    """Reflect a Cartesian field's transverse pattern through a local axis.

    Parameters
    ----------
    space : solprop.grid.FreeGrid
        Cartesian representation.
    spectrum : numpy.ndarray
        Stored spectrum, shape ``(n_modes, n_components, n_omega)``.
    image : numpy.ndarray
        The 2x2 position map, ``diag(-1, 1)`` or ``diag(1, -1)`` to rounding.

    Returns
    -------
    numpy.ndarray
        The reflected spectrum.

    Raises
    ------
    CouplingError
        If the map is not a reflection through x or y, or the reflected axis has
        an odd sample count and so is not symmetric about zero.
    """
    flips = np.round(np.diag(image))
    if np.max(np.abs(image - np.diag(flips))) > _FRAME_TOLERANCE:
        msg = (
            "this mirror image is a reflection through an axis oblique to the grid; "
            "align the s axis with the local x or y axis"
        )
        raise CouplingError(msg)
    shape = (space.n_y, space.n_x, *spectrum.shape[1:])
    samples = _to_field(space, spectrum).reshape(shape)
    for axis, count, flip in ((1, space.n_x, flips[0]), (0, space.n_y, flips[1])):
        if flip < 0:
            if count % 2:
                msg = (
                    "an odd sample count places no sample at zero, so the mirror "
                    "image is not on the grid; use an even count"
                )
                raise CouplingError(msg)
            samples = np.take(samples, (-np.arange(count)) % count, axis=axis)
    return _to_spectrum(space, samples.reshape(spectrum.shape))


@dataclass(frozen=True, eq=False)
class PlanarInterface:
    r"""A plane boundary between two lossless media, at normal incidence.

    Tangential E/H continuity gives :math:`t_E=2n_1/(n_1+n_2)` and
    :math:`r_E=(n_1-n_2)/(n_1+n_2)` per frequency, with
    :math:`T=(n_2/n_1)|t_E|^2` and :math:`R+T=1`. The factor is applied to every
    angular column, and the departure of an oblique column's flux from the
    normal-incidence coefficient is reported as approximation error. The output
    plane is in `second`, so a second application of the same interface is
    refused: a Fresnel factor is applied once.

    Attributes
    ----------
    first : solprop.spatial.SpatialMedium
        The incident medium; must be the plane's medium.
    second : solprop.spatial.SpatialMedium
        The transmitted medium.
    """

    first: SpatialMedium
    second: SpatialMedium

    def apply(self, state: SpatialState) -> ElementResult:
        """Transmit the plane through the interface.

        Parameters
        ----------
        state : SpatialState
            The input plane, in `first`.

        Returns
        -------
        ElementResult
            The transmitted plane, in `second`, and the report.

        Raises
        ------
        CouplingError
            If the plane is not in `first` or either medium is lossy.
        """
        grid = state.grid
        if not _same_medium(state.medium, self.first, grid):
            msg = (
                "the plane is not in the interface's first medium; a Fresnel factor "
                "is applied once, at the declared interface"
            )
            raise CouplingError(msg)
        band = _band(grid)
        first = _index(self.first, grid)[band]
        second = _index(self.second, grid)[band]
        for values in (first, second):
            if np.any(np.abs(np.imag(values)) > 1e-12 * np.real(values)):
                msg = "the normal-incidence interface model is for lossless media"
                raise CouplingError(msg)
        n_1, n_2 = np.real(first), np.real(second)
        transmission = 2 * n_1 / (n_1 + n_2)
        reflectance = np.zeros(grid.n_omega)
        reflectance[band] = ((n_1 - n_2) / (n_1 + n_2)) ** 2
        factor = np.ones(grid.n_omega)
        factor[band] = transmission
        after = state.with_spectrum(
            factor[None, None, :] * state.spectrum, medium=self.second
        )
        defect = (
            float(np.max(np.abs(reflectance[band] + (n_2 / n_1) * transmission**2 - 1)))
            if n_1.size
            else 0.0
        )
        return _uniform_result(
            state,
            after,
            element="PlanarInterface(normal incidence)",
            reflected=reflectance,
            extras=_Extras(diagnostics={"fresnel_defect": defect}),
        )


# --------------------------------------------------------------------- spectral optics


def _known_calibration(  # noqa: PLR0911 - one return per known element type
    transfer: TransferFunction,
) -> tuple[SpectralCalibration, ...] | None:
    """Return the affine-phase record of a known optics element.

    Parameters
    ----------
    transfer : solprop.optics.TransferFunction
        The element.

    Returns
    -------
    tuple of SpectralCalibration or None
        One record per factor (a chain gives one per member), or None for an
        unknown type.
    """
    if isinstance(transfer, Chain):
        records: list[SpectralCalibration] = []
        for member in transfer.elements:
            known = _known_calibration(member)
            if known is None:
                return None
            records.extend(known)
        return tuple(records)
    if isinstance(transfer, TaylorPhase):
        return (
            SpectralCalibration(
                False,
                False,
                transfer.wavelength,
                "TaylorPhase applies its coefficients as given",
            ),
        )
    if isinstance(transfer, Material):
        removes = transfer.wavelength is not None
        return (
            SpectralCalibration(
                False,
                removes,
                transfer.wavelength,
                "Material keeps beta0"
                + (" and removes the group delay at its wavelength" if removes else ""),
            ),
        )
    if isinstance(transfer, ChirpedMirror):
        return (
            SpectralCalibration(
                True,
                True,
                None,
                "ChirpedMirror data lack constant phase and delay",
            ),
        )
    if isinstance(transfer, GratingPair | PrismPair):
        return (
            SpectralCalibration(
                True,
                True,
                transfer.wavelength,
                f"{type(transfer).__name__} removes phase and group delay",
            ),
        )
    return None


def _phase_only(transfer: TransferFunction) -> bool | None:
    """Return whether a known optics model omits all loss.

    Parameters
    ----------
    transfer : solprop.optics.TransferFunction
        The element.

    Returns
    -------
    bool or None
        True for a phase-only model, False for one with loss, None if unknown.
    """
    if isinstance(transfer, Chain):
        members = [_phase_only(member) for member in transfer.elements]
        if any(member is None for member in members):
            return None
        return all(members)
    if isinstance(transfer, TaylorPhase | GratingPair | PrismPair):
        return True
    if isinstance(transfer, Material):
        return not transfer.loss
    if isinstance(transfer, ChirpedMirror):
        return transfer.reflections < 0 or transfer.amplitude.size == 0
    return None


def _reconstruction(transfer: TransferFunction) -> bool:
    """Return whether a known optics element is a reconstruction operation.

    Parameters
    ----------
    transfer : solprop.optics.TransferFunction
        The element.

    Returns
    -------
    bool
        True for a negative material length or an inverse mirror phase, in the
        element or any member of a chain.
    """
    if isinstance(transfer, Chain):
        return any(_reconstruction(member) for member in transfer.elements)
    if isinstance(transfer, Material):
        return transfer.thickness < 0
    if isinstance(transfer, ChirpedMirror):
        return transfer.reflections < 0
    return False


@dataclass(frozen=True, eq=False)
class SpectralTransfer:
    """A frequency-only transfer function acting coherently on every sample.

    Adapts any :class:`solprop.optics.TransferFunction` -- ``Material``,
    ``ChirpedMirror``, ``GratingPair``, ``PrismPair``, ``TaylorPhase`` -- an
    :class:`solprop.optics.Chain` as one block whose factors commute among
    themselves only, or a compressor's response described by its callable and
    metadata. The response crosses to the stored spectrum through
    :func:`solprop.optics.stored_factor`. Losses are kept per frequency.

    Attributes
    ----------
    transfer : solprop.optics.TransferFunction
        The optics-convention response ``H(omega)``.
    efficiency : float, callable or None
        Power efficiency per frequency in [0, 1], applied as ``sqrt(efficiency)``
        on top of a **phase-only** model. None applies none.
    calibration : SpectralCalibration, tuple of SpectralCalibration or None
        The affine phase the element removes. None derives it from a known type
        and is refused for any other callable.
    predict_loss : bool
        Ask for a lossy prediction: refused for a phase-only model without an
        explicit `efficiency`. Default False.
    reconstruction : bool
        Declare a reconstruction (inverse) operation; negative material lengths and
        inverse mirror phases are detected and flagged regardless. Default False.
    """

    transfer: TransferFunction
    efficiency: float | SpectralFunction | None = None
    calibration: SpectralCalibration | tuple[SpectralCalibration, ...] | None = None
    predict_loss: bool = False
    reconstruction: bool = False

    def _calibrations(self) -> tuple[SpectralCalibration, ...]:
        """Return the affine-phase records, derived or given.

        Returns
        -------
        tuple of SpectralCalibration
            One record per factor.

        Raises
        ------
        CouplingError
            If an unknown callable carries no calibration.
        """
        if isinstance(self.calibration, SpectralCalibration):
            return (self.calibration,)
        if self.calibration is not None:
            return tuple(self.calibration)
        known = _known_calibration(self.transfer)
        if known is None:
            msg = (
                f"{type(self.transfer).__name__} is not a known optics element: give "
                "calibration=SpectralCalibration(...) saying which affine phase it "
                "removes"
            )
            raise CouplingError(msg)
        return known

    def apply(self, state: SpatialState) -> ElementResult:
        """Apply the transfer function.

        Parameters
        ----------
        state : SpatialState
            The input plane.

        Returns
        -------
        ElementResult
            The output plane and its report, with the calibration records.

        Raises
        ------
        CouplingError
            If a lossy prediction lacks its efficiency, an efficiency is given to a
            lossy model, an efficiency is outside [0, 1], or the calibration is
            missing.
        """
        calibrations = self._calibrations()
        phase_only = _phase_only(self.transfer)
        grid = state.grid
        band = _band(grid)
        factor = stored_factor(grid, self.transfer)
        if phase_only is None:
            occupied = np.any(state.spectrum != 0, axis=(0, 1))
            phase_only = bool(np.all(np.abs(np.abs(factor[occupied]) - 1) <= 8 * _EPS))
        if self.efficiency is not None:
            if not phase_only:
                msg = (
                    "the transfer function already carries loss; an efficiency on "
                    "top of it would count the loss twice"
                )
                raise CouplingError(msg)
            efficiency = np.real(
                _spectral_values(self.efficiency, grid.omega[band], "efficiency")
            )
            if np.any((efficiency < 0) | (efficiency > 1)):
                msg = "an efficiency is a power fraction in [0, 1]"
                raise CouplingError(msg)
            factor = factor.copy()
            factor[band] = factor[band] * np.sqrt(efficiency)
        elif self.predict_loss and phase_only:
            msg = (
                f"{type(self.transfer).__name__} is a phase-only model: a lossy "
                "prediction needs an explicit efficiency="
            )
            raise CouplingError(msg)
        if not np.all(np.isfinite(factor)) or np.max(np.abs(factor)) > 1 + 8 * _EPS:
            msg = "a passive transfer function is finite with |H| <= 1"
            raise CouplingError(msg)
        after = state.with_spectrum(factor[None, None, :] * state.spectrum)
        notes: tuple[str, ...] = ()
        if isinstance(self.transfer, Chain):
            notes = tuple(
                f"factor {number}: {member!r}"
                for number, member in enumerate(self.transfer.elements)
            )
        return _uniform_result(
            state,
            after,
            element=f"SpectralTransfer({type(self.transfer).__name__})",
            removed_bins=band & (factor == 0),
            extras=_Extras(
                calibration=calibrations,
                reconstruction=self.reconstruction or _reconstruction(self.transfer),
                notes=notes,
            ),
        )


# --------------------------------------------------------------------- free space


@dataclass(frozen=True, eq=False)
class FreeSpace:
    r"""Homogeneous propagation over a distance, :math:`e^{+ik_zL}` per frequency.

    :math:`k_z=\sqrt{(\omega n/c)^2-\kappa^2}` at each absolute frequency and
    transverse wave number, on the order-zero Hankel columns of a radial grid or the
    DFT columns of a Cartesian one. An affine frame term is removed from the whole
    field, :math:`e^{-i(\phi_f+\omega\tau_f)}`, so that the pulse stays in its
    window; the report records ``delay_removed`` and ``phase_removed`` and
    :func:`solprop.coupling.transport` adds them to the plane's reference.

    Without `output` the field stays on its grid, propagated exactly in its
    periodic (Cartesian) or Dirichlet (radial) representation. A column at angle
    :math:`\theta` moves its rays by :math:`L\tan\theta`; content whose rays leave
    the window would wrap (or reflect), which is the same condition as sampling the
    propagator's phase :math:`k_zL` below its Nyquist rate in :math:`\kappa`. Content
    beyond ``sampling_tolerance`` is refused, unless ``beyond_window='remove'``
    declares its removal, which is then reported as removed energy. With `output`,
    a plane of another radius or sampling is evaluated by chunked direct summation;
    the method is always the caller's explicit choice:

    - ``output_method='angular'`` sums the propagated source spectrum at the output
      nodes (:math:`\sum_j v_jH_jJ_0(k_j\rho)/2\pi` or the trigonometric
      interpolant), exact and nonparaxial, inside the source window only; the field
      must stay away from the source window's edge.
    - ``output_method='fresnel'`` evaluates the Fresnel integral of the source
      samples (zero outside the source window) at the output nodes with the
      paraxial kernel, radial
      :math:`\frac{k}{iL}e^{ikL}e^{ik\rho^2/2L}\int E\,e^{ikr^2/2L}J_0(k\rho r/L)r\,dr`,
      Cartesian separably. It reaches any observation window; its paraxial phase
      bound :math:`k(\rho_{\max}+r_{\rm occ})^4/(8L^3)` is reported, and an
      undersampled kernel is refused.

    Significant evanescent content is refused rather than capped, unless an explicit
    `band_limit` removes it, with the removed energy reported.

    Attributes
    ----------
    distance : float
        Propagation distance in m, finite; negative back-propagates and is flagged
        as a reconstruction.
    medium : solprop.spatial.SpatialMedium
        The homogeneous medium of the gap; must be the plane's medium (a change of
        medium is a :class:`PlanarInterface`).
    output : solprop.grid.RadialGrid, solprop.grid.FreeGrid or None
        Output plane of the same kind as the source; None keeps the source grid.
    output_method : {'angular', 'fresnel'}
        How an `output` plane is evaluated. Default ``'angular'``.
    band_limit : float or None
        Largest retained propagation angle in rad, in (0, pi/2]; columns beyond it,
        and every nonpropagating column, are removed and reported. None removes
        nothing.
    frame_delay : float or None
        Delay :math:`\tau_f` in s removed from the whole field; None removes the
        group delay of the medium at the grid's reference frequency,
        ``beta1(omega0) L``. :func:`solprop.coupling.transport` fills it from the
        plane's frame when it is given one.
    frame_phase : float
        Constant phase :math:`\phi_f` in rad removed with the delay. Default 0.
    evanescent_tolerance : float or None
        Largest admitted reference-measure share of nonpropagating content; None is
        the rounding budget of the transforms that formed the field.
    sampling_tolerance : float
        Largest admitted reference-measure share of content whose rays leave the
        window (same-grid and angular routes), of field at the source edge
        (angular output) and of aliased content (Fresnel output). Default 1e-6.
    beyond_window : {'refuse', 'remove'}
        What to do with content whose rays leave the window over the distance:
        refuse it (the default), or remove it explicitly as an angular band,
        reported as ``removed`` with ``window_removed`` in the diagnostics.

    Examples
    --------
    >>> from solprop.coupling import FreeSpace
    >>> from solprop.spatial import SpatialMedium
    >>> FreeSpace(0.5, SpatialMedium(), beyond_window="remove").beyond_window
    'remove'
    """

    distance: float
    medium: SpatialMedium
    output: RadialGrid | FreeGrid | None = None
    output_method: Literal["angular", "fresnel"] = "angular"
    band_limit: float | None = None
    frame_delay: float | None = None
    frame_phase: float = 0.0
    evanescent_tolerance: float | None = None
    sampling_tolerance: float = 1e-6
    beyond_window: Literal["refuse", "remove"] = "refuse"

    def _validate(self, state: SpatialState) -> None:
        """Check the element against the plane.

        Parameters
        ----------
        state : SpatialState
            The input plane.

        Raises
        ------
        CouplingError
            If the distance, method, band limit or output plane is invalid, or
            the gap's medium is not the plane's.
        """
        if not (math.isfinite(self.distance) and math.isfinite(self.frame_phase)):
            msg = (
                f"distance and frame_phase must be finite, got {self.distance!r} and "
                f"{self.frame_phase!r}"
            )
            raise CouplingError(msg)
        if self.frame_delay is not None and not math.isfinite(self.frame_delay):
            msg = f"frame_delay must be finite, got {self.frame_delay!r}"
            raise CouplingError(msg)
        if self.beyond_window not in ("refuse", "remove"):
            msg = f"beyond_window is 'refuse' or 'remove', got {self.beyond_window!r}"
            raise CouplingError(msg)
        if self.output_method not in ("angular", "fresnel"):
            msg = f"output_method is 'angular' or 'fresnel', got {self.output_method!r}"
            raise CouplingError(msg)
        if self.band_limit is not None and not 0 < self.band_limit <= math.pi / 2:
            msg = f"band_limit is an angle in (0, pi/2], got {self.band_limit!r}"
            raise CouplingError(msg)
        if not _same_medium(state.medium, self.medium, state.grid):
            msg = (
                "the gap's medium is not the plane's: a change of medium is a "
                "PlanarInterface, applied once"
            )
            raise CouplingError(msg)
        if self.output is not None and (
            isinstance(self.output, LineGrid)
            or isinstance(self.output, RadialGrid)
            != isinstance(state.space, RadialGrid)
        ):
            msg = "the output plane must be the same kind of grid as the source"
            raise CouplingError(msg)

    def _delay(self, state: SpatialState) -> float:
        """Return the delay removed from the whole field.

        Parameters
        ----------
        state : SpatialState
            The input plane.

        Returns
        -------
        float
            ``frame_delay``, or ``beta1(omega0) L`` in s.
        """
        if self.frame_delay is not None:
            return self.frame_delay
        reference = np.array([state.grid.omega0])
        index = float(np.real(self.medium.index(reference))[0])
        slope = float(np.real(self.medium.dindex_domega(reference))[0])
        return (index + reference[0] * slope) / C * self.distance

    def apply(self, state: SpatialState) -> ElementResult:
        """Propagate the plane.

        Parameters
        ----------
        state : SpatialState
            The input plane.

        Returns
        -------
        ElementResult
            The output plane and the report, whose diagnostics include
            ``delay_removed`` in s and ``phase_removed`` in rad.

        Raises
        ------
        CouplingError
            If the element is invalid, the medium is not the plane's, the field
            holds significant evanescent content, its rays wrap beyond the window,
            the angular output leaves the source window, or the Fresnel kernel is
            undersampled.
        """
        self._validate(state)
        delay = self._delay(state)
        flux = _flux(state)
        measured_in = _measure(state, flux)
        kept, removed = self._limited(state, flux, measured_in)
        if self.output is not None and self.output_method == "fresnel":
            return self._fresnel(state, kept, measured_in, removed, delay)
        propagated, absorbed, window_removed, diagnostics = self._propagate(
            kept, flux, delay
        )
        removed += window_removed
        notes: tuple[str, ...] = ()
        result = propagated
        if self.output is not None:
            edge = _edge_share(propagated)
            if edge > self.sampling_tolerance:
                msg = (
                    f"the propagated field puts {edge:.3e} of itself at the source "
                    "window's edge, and the angular output treats the source as zero "
                    "beyond it: enlarge the source window, or evaluate this plane "
                    'with output_method="fresnel", which reaches any window'
                )
                raise CouplingError(msg)
            result = propagated.with_spectrum(
                _angular_output(propagated, self.output), space=self.output
            )
            diagnostics["output_edge_share"] = _edge_share(result)
            notes = ("output plane by direct angular-spectrum summation",)
        diagnostics["delay_removed"] = delay
        diagnostics["phase_removed"] = self.frame_phase
        measured_out = _measure(result, _flux(result))
        report = _report(
            f"FreeSpace(L={self.distance:g} m)",
            measured_in,
            measured_out,
            specified_loss=absorbed,
            removed=removed,
            numerical_error=(
                0.0 if self.beyond_window == "remove" else diagnostics["wrap_share"]
            )
            * measured_in.energy,
            extras=_Extras(
                reconstruction=self.distance < 0, diagnostics=diagnostics, notes=notes
            ),
        )
        return ElementResult(result, report)

    def _limited(
        self, state: SpatialState, flux: SpatialFlux, measured: _Measured
    ) -> tuple[SpatialState, float]:
        """Refuse evanescent content, or remove the declared angular band.

        Parameters
        ----------
        state : SpatialState
            The input plane.
        flux : solprop.spatial.SpatialFlux
            Its measure.
        measured : _Measured
            Its energy.

        Returns
        -------
        tuple
            The retained plane and the physical energy removed, J.

        Raises
        ------
        CouplingError
            If evanescent content exceeds its tolerance, or any remains for a
            backward propagation.
        """
        content = _angular_content(state, flux, 1.0)
        if self.band_limit is None:
            tolerance = self.evanescent_tolerance
            if tolerance is None:
                tolerance = 8 * state.space.n_columns * state.grid.n_omega * _EPS
            share = 0.0 if content.total == 0 else content.evanescent / content.total
            if share > tolerance or (self.distance < 0 and content.evanescent > 0):
                msg = (
                    f"the field puts {share:.3e} of its reference measure into "
                    "nonpropagating columns, which no forward plane carries; remove "
                    "them explicitly with band_limit= or widen the beam"
                )
                raise CouplingError(msg)
            return state, 0.0
        index = np.real(_index(state.medium, state.grid))
        reach = (index * state.grid.omega / C * math.sin(self.band_limit)) ** 2
        squares = _transverse_squares(state.space)
        keep = (squares[:, None] <= reach[None, :]) & _band(state.grid)[None, :]
        # A nonpropagating column is never kept, whatever the angle: sin <= 1.
        kept = state.with_spectrum(keep[:, None, :] * state.spectrum)
        return kept, measured.energy - spatial_energy(kept.spectrum, flux)

    def _wavenumber(
        self, state: SpatialState, columns: slice
    ) -> Complex[np.ndarray, "n_chunk n_omega"]:
        """Return the longitudinal wave number of a chunk of columns.

        Parameters
        ----------
        state : SpatialState
            The plane.
        columns : slice
            Transverse columns.

        Returns
        -------
        numpy.ndarray
            :math:`k_z` in m^-1 with nonnegative imaginary part for a lossless
            medium (decay of a nonpropagating column), shape ``(n_chunk, n_omega)``;
            zero off band.
        """
        grid = state.grid
        band = _band(grid)
        square = np.zeros(grid.n_omega, dtype=np.complex128)
        square[band] = (grid.omega[band] * _index(state.medium, grid)[band] / C) ** 2
        longitudinal = square[None, :] - _transverse_squares(state.space)[columns, None]
        # +0j keeps the principal root's branch at +i|k_z| on the cutoff side.
        root = np.sqrt(longitudinal + 0j)
        return np.where(band[None, :], root, 0.0)

    def _propagate(
        self, state: SpatialState, flux: SpatialFlux, delay: float
    ) -> tuple[SpatialState, float, float, dict[str, float]]:
        """Propagate the stored spectrum exactly on its own grid.

        Parameters
        ----------
        state : SpatialState
            The (band-limited) input plane.
        flux : solprop.spatial.SpatialFlux
            Its measure.
        delay : float
            Delay removed from the whole field, s.

        Returns
        -------
        tuple
            The propagated plane, the energy absorbed by the medium in J, the
            energy removed beyond the window in J, and the wrap diagnostics.

        Raises
        ------
        CouplingError
            If content whose rays leave the window exceeds ``sampling_tolerance``
            and its removal is not declared.
        """
        grid = state.grid
        band = _band(grid)
        wrap = _WrapCount()
        beyond = np.zeros((state.space.n_columns, grid.n_omega), dtype=bool)
        for columns in flux.chunks():
            k_z = self._wavenumber(state, columns)
            beyond[columns] = wrap.add(state, columns, k_z, self.distance)
        share = wrap.share()
        window_removed = 0.0
        source = state
        if self.beyond_window == "remove":
            source = state.with_spectrum(~beyond[:, None, :] * state.spectrum)
            window_removed = spatial_energy(state.spectrum, flux) - spatial_energy(
                source.spectrum, flux
            )
        elif share > self.sampling_tolerance:
            msg = (
                f"{share:.3e} of the field travels beyond the window over "
                f"{self.distance:g} m (largest ray shift {wrap.max_ratio:.3g} "
                "half-widths) and would wrap: the propagator's phase is "
                "undersampled there. Enlarge the window"
                + (
                    ', evaluate this output plane with output_method="fresnel", '
                    "which reaches any window,"
                    if self.output is not None
                    else ", evaluate a larger output plane with output_method="
                    '"fresnel",'
                )
                + " or declare the angular band removal with beyond_window='remove'"
            )
            raise CouplingError(msg)
        spectrum = np.array(source.spectrum)
        attenuated = np.array(source.spectrum)
        for columns in flux.chunks():
            k_z = self._wavenumber(state, columns)
            # Optics exp(+i(k_z L - phi - omega tau)) crosses as its conjugate.
            exponent = -1j * np.conj(k_z) * self.distance + 1j * (
                grid.omega * delay + self.frame_phase
            )
            factor = np.where(band[None, :], np.exp(exponent), 1.0)
            spectrum[columns] *= factor[:, None, :]
            attenuated[columns] *= np.abs(factor)[:, None, :]
        absorbed = spatial_energy(source.spectrum, flux) - spatial_energy(
            attenuated, flux
        )
        if np.all(np.imag(_index(state.medium, grid))[band] == 0):
            absorbed = 0.0
        propagated = state.with_spectrum(spectrum)
        diagnostics = {
            "wrap_share": share,
            "max_ray_shift_ratio": wrap.max_ratio,
            "edge_share": _edge_share(propagated),
            "window_removed": window_removed,
        }
        return propagated, absorbed, window_removed, diagnostics

    def _fresnel(
        self,
        state: SpatialState,
        kept: SpatialState,
        measured_in: _Measured,
        removed: float,
        delay: float,
    ) -> ElementResult:
        """Evaluate the Fresnel integral of the source samples at the output nodes.

        Parameters
        ----------
        state : SpatialState
            The input plane, for the medium and grid.
        kept : SpatialState
            The band-limited input plane.
        measured_in : _Measured
            Energy of the input plane.
        removed : float
            Energy removed by the band limit, J.
        delay : float
            Delay removed from the whole field, s.

        Returns
        -------
        ElementResult
            The output plane and the report.

        Raises
        ------
        CouplingError
            If the distance is not positive or the kernel is undersampled on the
            source grid.
        """
        output = self.output
        assert output is not None
        if self.distance <= 0:
            msg = "the Fresnel output needs a positive distance"
            raise CouplingError(msg)
        grid = state.grid
        band = _band(grid)
        index = _index(state.medium, grid)
        samples = _to_field(kept.space, kept.spectrum)
        real_index = np.real(index)
        occupied = _occupied(_node_content(kept.space, grid, real_index, samples))
        source_x, source_y = _nodes(kept.space)
        source_radius = np.hypot(source_x, source_y)
        reach = float(np.max(source_radius[occupied], initial=0.0))
        out_x, out_y = _nodes(output)
        out_reach = float(np.max(np.hypot(out_x, out_y)))
        wavenumber = real_index * grid.omega / C
        k_max = float(np.max(wavenumber[band], initial=0.0))
        spacing = _spacing(kept.space)
        # The integrand's local spatial frequency is the source's own transverse
        # wave number plus the kernel's chirp; content whose sum passes the nodes'
        # Nyquist rate aliases in the node rule.
        chirp = k_max * (reach + out_reach) / self.distance
        step = chirp * spacing
        aliased = _share_beyond(kept, real_index, math.pi / spacing - chirp)
        if aliased > self.sampling_tolerance:
            msg = (
                f"the Fresnel kernel's phase changes by {step:.3g} rad between source "
                f"nodes, and {aliased:.3e} of the field would alias with it; refine "
                "the source grid or move the plane further"
            )
            raise CouplingError(msg)
        bound = k_max * (reach + out_reach) ** 4 / (8 * self.distance**3)
        field_out = np.zeros(
            (output.n_columns, kept.n_pol, grid.n_omega), dtype=np.complex128
        )
        for bin_index in np.flatnonzero(band):
            k = complex(index[bin_index]) * grid.omega[bin_index] / C
            field_out[:, :, bin_index] = _fresnel_bin(
                kept.space, output, samples[:, :, bin_index], k, self.distance
            ) * np.exp(1j * (grid.omega[bin_index] * delay + self.frame_phase))
        result = kept.with_spectrum(_to_spectrum(output, field_out), space=output)
        measured_out = _measure(result, _flux(result))
        absorbed = float(
            np.sum(
                measured_in.spectrum
                * (1 - np.exp(-2 * np.imag(index) * grid.omega / C * self.distance))
            )
        )
        report = _report(
            f"FreeSpace(L={self.distance:g} m, Fresnel output)",
            measured_in,
            measured_out,
            specified_loss=absorbed,
            removed=removed,
            extras=_Extras(
                diagnostics={
                    "fresnel_phase_bound": bound,
                    "kernel_phase_step": step,
                    "kernel_alias_share": aliased,
                    "output_edge_share": _edge_share(result),
                    "delay_removed": delay,
                    "phase_removed": self.frame_phase,
                },
                notes=("paraxial Fresnel kernel from the source samples",),
            ),
        )
        return ElementResult(result, report)


class _WrapCount:
    """Accumulate the reference-measure share whose rays leave the window."""

    def __init__(self) -> None:
        """Start an empty count."""
        self.total = 0.0
        self.wrapped = 0.0
        self.max_ratio = 0.0

    def add(
        self,
        state: SpatialState,
        columns: slice,
        k_z: Complex[np.ndarray, "n_chunk n_omega"],
        distance: float,
    ) -> Bool[np.ndarray, "n_chunk n_omega"]:
        """Add a chunk of columns and return which of their bins leave the window.

        A column at angle theta moves its rays by ``L tan(theta) = L kappa / k_z``;
        beyond the window's half-width the periodic (Cartesian) or reflecting
        (radial) representation folds them back.

        Parameters
        ----------
        state : SpatialState
            The plane.
        columns : slice
            Transverse columns.
        k_z : numpy.ndarray
            Their longitudinal wave numbers, shape ``(n_chunk, n_omega)``.
        distance : float
            Propagation distance in m.

        Returns
        -------
        numpy.ndarray
            True where a propagating column's rays move beyond the window's
            half-width, shape ``(n_chunk, n_omega)``.
        """
        grid = state.grid
        space = state.space
        weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
        index = np.real(_index(state.medium, grid))
        density = np.sum(np.abs(state.spectrum[columns]) ** 2, axis=1)
        content = space.spectral_weights[columns, None] * (weights * index) * density
        real = np.real(k_z)
        if isinstance(space, RadialGrid):
            ratios = [space.k_r[columns][:, None] / space.radius]
        else:
            k_x = np.tile(space.k_x, space.n_y)[columns]
            k_y = np.repeat(space.k_y, space.n_x)[columns]
            half_x = space.n_x * float(space.x[1] - space.x[0]) / 2
            half_y = space.n_y * float(space.y[1] - space.y[0]) / 2
            ratios = [np.abs(k_x)[:, None] / half_x, np.abs(k_y)[:, None] / half_y]
        ratio = np.zeros(content.shape)
        positive = real > 0
        for part in ratios:
            shift = np.divide(
                abs(distance) * np.broadcast_to(part, real.shape),
                real,
                out=np.zeros(real.shape),
                where=positive,
            )
            ratio = np.maximum(ratio, shift)
        occupied = content > _OCCUPIED * float(np.max(content, initial=0.0))
        self.total += float(np.sum(content))
        self.wrapped += float(np.sum(content[ratio > 1]))
        if np.any(occupied):
            self.max_ratio = max(self.max_ratio, float(np.max(ratio[occupied])))
        return ratio > 1

    def share(self) -> float:
        """Return the wrapped share.

        Returns
        -------
        float
            Share in [0, 1]; zero for a zero field.
        """
        return 0.0 if self.total == 0 else self.wrapped / self.total


def _share_beyond(
    state: SpatialState, index: Float[np.ndarray, " n_omega"], limit: float
) -> float:
    """Return the reference-measure share in columns beyond a transverse wave number.

    Parameters
    ----------
    state : SpatialState
        The plane.
    index : numpy.ndarray
        Real index per bin, zero off band.
    limit : float
        Transverse wave number in rad/m; a nonpositive limit counts every column.

    Returns
    -------
    float
        Share in [0, 1]; zero for a zero field.
    """
    grid = state.grid
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    density = np.sum(np.abs(state.spectrum) ** 2, axis=1)
    content = state.space.spectral_weights * np.sum((weights * index) * density, axis=1)
    total = float(np.sum(content))
    if total == 0:
        return 0.0
    beyond = _transverse_squares(state.space) > max(limit, 0.0) ** 2
    if limit <= 0:
        beyond[:] = True
    return float(np.sum(content[beyond])) / total


def _spacing(space: RadialGrid | FreeGrid) -> float:
    """Return the largest spacing between neighbouring nodes.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Transverse representation.

    Returns
    -------
    float
        Spacing in m.
    """
    if isinstance(space, RadialGrid):
        return float(np.max(np.diff(np.concatenate([[0.0], space.r]))))
    return max(float(space.x[1] - space.x[0]), float(space.y[1] - space.y[0]))


def _angular_output(
    state: SpatialState, output: RadialGrid | FreeGrid
) -> Complex[np.ndarray, "n_out n_pol n_omega"]:
    r"""Sum a stored spectrum at another grid's nodes and transform it there.

    Radial: :math:`E(\rho)=\sum_jv_jH_jJ_0(k_j\rho)/2\pi`, the QDHT inverse
    evaluated off its nodes. Cartesian: the trigonometric interpolant with its
    first-sample origin, the Nyquist column split symmetrically. Nodes outside the
    source window are zero. The kernel is built in blocks of output nodes; no
    transverse-by-transverse-by-frequency array is formed.

    Parameters
    ----------
    state : SpatialState
        The source plane.
    output : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The output representation, of the same kind.

    Returns
    -------
    numpy.ndarray
        The output's stored spectrum, shape ``(n_out, n_pol, n_omega)``.
    """
    space = state.space
    spectrum = state.spectrum
    flat = spectrum.reshape(spectrum.shape[0], -1)
    if isinstance(space, RadialGrid):
        assert isinstance(output, RadialGrid)
        rho = np.asarray(output.r)
        samples = np.zeros((rho.size, flat.shape[1]), dtype=np.complex128)
        scale = space.spectral_weights / (2 * math.pi)
        block = max(1, _CHUNK // space.n_r)
        for start in range(0, rho.size, block):
            part = rho[start : start + block]
            kernel = j0(np.outer(part, space.k_r)) * scale[None, :]
            values = kernel @ flat
            samples[start : start + block] = np.where(
                (part < space.radius)[:, None], values, 0.0
            )
        field_out = samples.reshape(rho.size, *spectrum.shape[1:])
        return _to_spectrum(output, field_out)
    assert isinstance(output, FreeGrid)
    kernel_x = _interpolant(space.x, space.k_x, output.x)
    kernel_y = _interpolant(space.y, space.k_y, output.y)
    cube = spectrum.reshape(space.n_y, space.n_x, -1)
    n_inner = cube.shape[2]
    block = max(1, _CHUNK // max(1, space.n_y * output.n_x))
    result = np.zeros((output.n_y, output.n_x, n_inner), dtype=np.complex128)
    for start in range(0, n_inner, block):
        part = cube[:, :, start : start + block]
        along_x = np.einsum("yxp,xq->yqp", part, kernel_x)
        result[:, :, start : start + block] = np.einsum(
            "yqp,yr->rqp", along_x, kernel_y
        )
    field_out = result.reshape(output.n_columns, *spectrum.shape[1:]) / space.n_columns
    return _to_spectrum(output, field_out)


def _interpolant(
    axis: Float[np.ndarray, " n_source"],
    frequencies: Float[np.ndarray, " n_source"],
    target: Float[np.ndarray, " n_target"],
) -> Complex[np.ndarray, "n_source n_target"]:
    """Return the trigonometric interpolation kernel of one periodic axis.

    Parameters
    ----------
    axis : numpy.ndarray
        Source sample positions in m, uniform, shape ``(n_source,)``.
    frequencies : numpy.ndarray
        Their DFT angular frequencies in rad/m, FFT order.
    target : numpy.ndarray
        Output positions in m, shape ``(n_target,)``.

    Returns
    -------
    numpy.ndarray
        ``exp(i k (x' - x_0))``, zero for outputs outside the source period; the
        Nyquist row of an even axis is the cosine, so a real field stays real.
    """
    origin = float(axis[0])
    spacing = float(axis[1] - axis[0])
    offset = target - origin
    kernel = np.exp(1j * np.outer(frequencies, offset))
    count = axis.size
    if count % 2 == 0:
        kernel[count // 2] = np.cos(frequencies[count // 2] * offset)
    inside = (offset >= -spacing / 2) & (offset < count * spacing - spacing / 2)
    return kernel * inside[None, :]


def _fresnel_bin(
    source: RadialGrid | FreeGrid,
    output: RadialGrid | FreeGrid,
    samples: Complex[np.ndarray, "n_source n_pol"],
    wavenumber: complex,
    distance: float,
) -> Complex[np.ndarray, "n_target n_pol"]:
    r"""Evaluate the paraxial Fresnel integral at one frequency.

    Optics convention, crossed to the stored samples by conjugating the kernel:
    radial :math:`\frac{k}{iL}e^{ikL}e^{ik\rho^2/2L}\sum_m\frac{v_m}{2\pi}E_m
    e^{ikr_m^2/2L}J_0(k\rho r_m/L)`, Cartesian
    :math:`\frac{k}{2\pi iL}e^{ikL}\sum_nE_ne^{ik|\boldsymbol\rho-\mathbf r_n|^2/2L}
    \Delta A`, separable in x and y. The real part of ``k`` sets the phases and
    the imaginary part attenuates through ``exp(ikL)`` only.

    Parameters
    ----------
    source, output : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Source and output representations, of one kind.
    samples : numpy.ndarray
        Source field samples at one bin, shape ``(n_source, n_pol)``.
    wavenumber : complex
        :math:`k = \omega n / c` in m^-1.
    distance : float
        Positive distance in m.

    Returns
    -------
    numpy.ndarray
        Output samples at that bin, shape ``(n_target, n_pol)``.
    """
    k = wavenumber.real
    prefactor = (wavenumber / (1j * distance)) * np.exp(1j * wavenumber * distance)
    common = np.conj(prefactor)
    if isinstance(source, RadialGrid):
        assert isinstance(output, RadialGrid)
        rho = np.asarray(output.r)
        weights = source.radial_weights / (2 * math.pi)
        chirp_in = np.exp(-1j * k * source.r**2 / (2 * distance))
        inner = samples * (weights * chirp_in)[:, None]
        values = np.zeros((rho.size, samples.shape[1]), dtype=np.complex128)
        block = max(1, _CHUNK // source.n_r)
        for start in range(0, rho.size, block):
            part = rho[start : start + block]
            kernel = j0(np.outer(part, source.r) * (k / distance))
            values[start : start + block] = kernel @ inner
        chirp = np.exp(-1j * k * rho**2 / (2 * distance))
        return common * chirp[:, None] * values
    assert isinstance(output, FreeGrid)
    assert isinstance(source, FreeGrid)
    phase = k / (2 * distance)
    kernel_x = np.exp(-1j * phase * np.subtract.outer(output.x, source.x) ** 2)
    kernel_y = np.exp(-1j * phase * np.subtract.outer(output.y, source.y) ** 2)
    area = float(source.spatial_weights[0])
    plane = samples.reshape(source.n_y, source.n_x, -1)
    values = np.einsum("ry,yxp,qx->rqp", kernel_y, plane, kernel_x, optimize=True)
    scale = common / (2 * math.pi) * area
    return scale * values.reshape(output.n_columns, -1)

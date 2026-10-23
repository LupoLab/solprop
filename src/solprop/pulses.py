# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Initial pulses on either grid: shape, exact scaling and the user's spectral phase.

Every builder here is a frozen dataclass with one method, ``build(grid)``, which returns
the initial spectrum in the canonical ``(n_modes, n_pol, n_omega)`` layout of
``docs/dev/conventions.md`` §6 with both leading axes equal to one. The spectrum is in
the internal FFT convention, ready to be handed to the translation layer.

**Signs.** Spectral phases, Taylor coefficients and delays are given in the **optics**
convention (``docs/dev/conventions.md`` §2): a positive group-delay dispersion
chirps the pulse positively, and a positive ``t0`` makes it arrive later. They are
converted to the internal convention exactly once, when the phase factor is formed, with
:func:`solprop.fourier.to_internal`; a measured *absolute* spectrum additionally needs
the half-window origin shift and goes through
:func:`solprop.fourier.spectrum_to_internal`. This module is the first of the three
boundary places where the optics sign convention meets the internal FFT convention
(see ``docs/dev/decisions.md``), and nothing else in it conjugates anything.

**Scaling.** A pulse is specified by its ``energy`` or by its ``peak_power``, never
both. Both are realised exactly on the grid, to the round-off of one rescaling, and
both are applied *after* the spectral phase, so the peak power of a chirped pulse is the
peak power the caller asked for.

The manual page is ``docs/guide/setup/pulses.md``.
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Complex, Float, Inexact
from scipy.interpolate import make_interp_spline

from solprop import fourier
from solprop._display import Displayed, engineering, line, phase_text, summarise
from solprop.constants import EPS0, C
from solprop.exceptions import PulseError
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.linear import Frame
from solprop.modes.slab import PlanarGrid, gaussian_projection
from solprop.optics import taylor_phase
from solprop.spatial import (
    FluxSide,
    SpatialDispersion,
    SpatialFlux,
    SpatialMedium,
    plane_sample,
    power_trace,
)

__all__ = [
    "CWPulse",
    "CustomPulse",
    "DataPulse",
    "GaussianPulse",
    "PlanarPulse",
    "Pulse",
    "PulseSum",
    "SechPulse",
    "ShapedPulse",
    "SpatialPulse",
]

logger = logging.getLogger(__name__)

# sech^2(x) falls to one half at x = ln(1 + sqrt(2)), so a sech^2 pulse of intensity
# FWHM tau has natural width tau/(2 ln(1 + sqrt(2))).
SECH_FWHM_TO_WIDTH = 1.0 / (2.0 * math.log(1.0 + math.sqrt(2.0)))

# cosh(340)**2 is 1.2e294, the largest such square that stays finite in double
# precision, and sech^2 there is 8e-295: zero for every purpose, but reached without an
# overflow warning, which is an error in the test session.
_SECH_CLIP = 340.0

# exp(-745) is the smallest positive double, so an exponent clipped there is already
# indistinguishable from zero; clipping the *argument* keeps a large super-Gaussian
# order from overflowing before the exponential can underflow.
_GAUSS_EXPONENT_CLIP = 745.0

# A cubic B-spline needs four points, so tabulated data shorter than this cannot be
# interpolated onto a grid.
_MIN_DATA_POINTS = 4

_Quantity = Literal["energy", "peak_power", "average_power"]


@runtime_checkable
class Pulse(Protocol):
    """The one method every input pulse provides.

    A pulse is a specification, not an array: it is built against a particular grid, and
    the same specification may be built against a field grid and an envelope grid to
    give the same physical pulse in the two representations.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import GaussianPulse, Pulse
    >>> pulse: Pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 2e-12)
    >>> pulse.build(grid).shape[:2]
    (1, 1)
    """

    def build(self, grid: TimeGrid) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the initial spectrum of this pulse on the given grid.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Grid to build on; its ``kind`` selects the field or envelope layout.

        Returns
        -------
        numpy.ndarray
            Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, complex, in the square
            root of watts and in the internal FFT convention.
        """
        ...


@dataclass(frozen=True)
class SpatialPulse(Displayed):
    """Wrap a temporal pulse in an energy-normalised transverse beam.

    Attributes
    ----------
    temporal : Pulse
        Existing temporal pulse shape. Its energy or peak power sets the whole beam's
        physical energy or peak axial power through the launch plane, in the medium
        there (SEP 34). On a
        :class:`solprop.grid.LineGrid` the beam is uniform in y and these are per
        metre of y: the energy is in J/m and the peak power in W/m
        (:func:`solprop.modes.slab.gaussian_to_per_unit_width` converts a beam of
        finite height).
    waist : float
        Positive finite 1/e^2 intensity radius in m; on a line, the 1/e^2
        intensity half-width in x.
    order : int
        Super-Gaussian order, at least 1. The field profile is
        ``exp(-(r/waist)**(2*order))``, with ``r = |x|`` on a line.
    propz : float
        Signed launch pre-propagation distance in m, finite. Applied by the exact
        longitudinal operator with the launch plane's material frozen. Zero launches
        at the waist; negative values launch before it. Normalisation follows the
        pre-propagation, so the launch plane carries the requested quantity.
    jones : tuple of complex
        One scalar component or two optics-convention Cartesian components (x, y).
        The vector is normalised to unit squared magnitude. Default is scalar.

    Raises
    ------
    PulseError
        If the waist, order, distance or Jones vector is invalid.
    """

    temporal: Pulse
    waist: float
    order: int = 1
    propz: float = 0.0
    jones: tuple[complex, ...] = (1.0 + 0.0j,)

    def __post_init__(self) -> None:
        """Validate the beam specification before sampling any grids."""
        _check_positive("waist", self.waist)
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, int)
            or self.order < 1
        ):
            msg = "spatial order must be an integer of at least 1"
            raise PulseError(msg)
        if not math.isfinite(self.propz):
            msg = "spatial propz must be finite"
            raise PulseError(msg)
        vector = np.asarray(self.jones, dtype=np.complex128)
        if (
            len(vector) not in (1, 2)
            or not np.all(np.isfinite(vector))
            or not np.any(vector)
        ):
            msg = (
                "spatial jones must contain one or two finite components "
                "with nonzero intensity"
            )
            raise PulseError(msg)

    def __repr__(self) -> str:
        """Return a one-line description: the beam around its temporal pulse.

        Returns
        -------
        str
            For example ``SpatialPulse(waist=100 um, order=1, GaussianPulse(...))``.
        """
        return line(
            "SpatialPulse",
            [
                ("waist", engineering(self.waist, "m")),
                ("order", self.order),
                ("", repr(self.temporal)),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "SpatialPulse: a super-Gaussian beam carrying a temporal pulse", [
            ("waist", f"{engineering(self.waist, 'm')} (1/e^2 intensity radius)"),
            ("order", self.order),
            ("launched", f"{engineering(self.propz, 'm')} from the waist"),
            ("components", len(self.jones)),
            ("temporal", str(self.temporal)),
        ]

    def build(
        self,
        grid: TimeGrid,
        space: RadialGrid | FreeGrid | PlanarGrid,
        *,
        medium: SpatialMedium,
        z: float = 0.0,
        side: FluxSide | None = None,
        frame: Frame | None = None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Build the beam's transverse and temporal spectrum on a spatial grid.

        The requested energy or peak power is the physical axial Poynting flux
        through the launch plane, in the medium there (SEP 34): what an energy
        meter in that medium reads. Each member of a :class:`PulseSum` carries its
        own requested quantity. The inactive band and zero frequency are
        hard-masked as everywhere; a beam with more than rounding-level content in
        nonpropagating columns is refused, since no energy can be assigned to it.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Field or envelope temporal grid.
        space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
            Transverse samples and their spectral integration measure. A
            :class:`solprop.modes.slab.PlanarGrid` is refused: its launch is a
            :class:`PlanarPulse`.
        medium : solprop.spatial.SpatialMedium
            The medium at the launch plane; there is no vacuum default.
        z : float, optional
            Position of the launch plane in m, at which the medium is evaluated.
        side : {'before', 'after'} or None, optional
            Side of a declared index jump at ``z``; required there.
        frame : solprop.linear.Frame or None, optional
            Launch frame, normally the run's frame. None uses the grid's co-moving
            frame without envelope THG.

        Returns
        -------
        numpy.ndarray
            Hankel/temporal spectrum in V m or Cartesian DFT/temporal spectrum
            in V/m, shape ``(n_modes, n_pol, n_omega)``, in the internal convention.

        Raises
        ------
        PulseError
            If the grid is a planar one, the temporal pulse is not scalar, the beam
            has no propagating content or more than rounding-level nonpropagating
            content, or launch pre-propagation produces a nonfinite field.
        """
        if isinstance(space, PlanarGrid):
            msg = (
                "a planar grid's launch is projected onto its slab modes in x: use "
                "PlanarPulse(temporal, waist_y, waist_x=...), not a SpatialPulse"
            )
            raise PulseError(msg)
        flux = SpatialFlux.resolve(grid, space, medium, z=z, side=side)
        limit = _GAUSS_EXPONENT_CLIP ** (1 / (2 * self.order))
        radius = np.minimum(space.r, self.waist * limit) / self.waist
        profile = np.exp(-(radius ** (2 * self.order)))
        spectrum = (
            space.hankel.forward(profile)
            if isinstance(space, RadialGrid)
            else np.fft.fft2(profile.reshape(space.n_y, space.n_x)).ravel()
        )
        jones = np.asarray(self.jones, dtype=np.complex128)
        jones = jones / max(
            float(np.max(np.abs(np.real(jones)))), float(np.max(np.abs(np.imag(jones))))
        )
        jones = jones / np.sqrt(np.sum(np.abs(jones) ** 2))
        live = np.concatenate(
            [flux.admittance(columns) != 0 for columns in flux.chunks()]
        )
        exponent = None
        if self.propz != 0:
            working_frame = Frame.comoving(grid) if frame is None else frame
            operator = SpatialDispersion(medium, grid, working_frame).operator(
                space.k_r**2 if isinstance(space, RadialGrid) else space,
                plane_sample(medium, z, side),
            )
            exponent = operator * self.propz
        total = np.zeros((spectrum.size, jones.size, grid.n_omega), complex)
        for member in _members(self.temporal):
            temporal = member.build(grid)
            if temporal.shape != (1, 1, grid.n_omega):
                msg = "SpatialPulse requires a scalar single-column temporal pulse"
                raise PulseError(msg)
            # A unit-peak profile keeps intermediate values near the final scale.
            field = (
                spectrum[:, None, None]
                * fourier.to_internal(jones)[None, :, None]
                * temporal
            )
            if exponent is not None:
                if np.max(np.real(exponent)) > math.log(np.finfo(float).max) - math.log(
                    max(1.0, float(np.max(np.abs(field))))
                ):
                    msg = "spatial pre-propagation would overflow the field"
                    raise PulseError(msg)
                field = field * np.exp(exponent)[:, None, :]
            field = _hard_masked(field, live, space.spectral_weights, grid)
            total += _normalised(member, temporal, field, flux, grid)
        if not np.all(np.isfinite(total)):
            msg = "spatial pre-propagation produced a nonfinite field"
            raise PulseError(msg)
        return np.asarray(total, dtype=np.complex128)


@dataclass(frozen=True)
class PlanarPulse(SpatialPulse):
    """Launch an astigmatic Gaussian beam into the hybrid planar grid.

    The beam is ``exp(-((x - offset_x)/waist_x)**2) exp(-(|y|/waist)**(2 order))`` in
    field, carrying the temporal pulse, polarised along the slab modes' own
    component (``y`` for TE, ``x`` for TM). In x it is projected onto the slab modes
    (:func:`solprop.modes.slab.gaussian_projection`, with signs); in y it is sampled
    on the grid and transformed. The temporal pulse's energy or peak power is the
    **incident** beam's, over the whole beam in x and the sampled y axis: what falls
    on the plates is lost at the entrance, so the modes carry the coupled share of it
    (98.8 % into ``p = 1`` at ``waist_x = 0.735 a``, Arnold et al. 2010), not a beam
    renormalised over the gap. The geometry integrates over y, so
    the energy is in J and the peak power in W.

    Attributes
    ----------
    temporal : Pulse
        Temporal pulse shape; its energy (J) or peak power (W) is the incident
        beam's.
    waist : float
        The 1/e **field** radius in y (the 1/e^2 intensity radius), m, positive.
    order : int
        Super-Gaussian order in y, at least 1.
    propz : float
        Must be 0: the launch is at the entrance plane; a free-space
        pre-propagation before a guide is refused.
    jones : tuple of complex
        One component, a phase; the component itself is the modes'. Two components
        are refused, the planar state being scalar.
    waist_x : float
        The 1/e field radius in x, m, positive (keyword).
    offset_x : float
        The beam centre's displacement from the mid-plane in x, m (keyword); an
        offset couples to even orders.

    Raises
    ------
    PulseError
        If a waist, the order, `offset_x`, `propz` or `jones` is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.modes import PlanarGrid, SlabMode
    >>> from solprop.pulses import GaussianPulse, PlanarPulse
    >>> grid = EnvelopeGrid.from_limits(0.1, 800e-9, (650e-9, 1000e-9), 400e-15)
    >>> space = PlanarGrid([SlabMode(63.5e-6, order=p) for p in (1, 3)], 1e-3, 32)
    >>> pulse = PlanarPulse(
    ...     GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
    ...     250e-6, waist_x=0.735 * 63.5e-6,
    ... )
    >>> pulse.build(grid, space).shape
    (64, 1, 256)
    """

    waist_x: float = field(kw_only=True)
    offset_x: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the beam and refuse what the planar launch does not carry."""
        super().__post_init__()
        _check_positive("waist_x", self.waist_x)
        if not math.isfinite(self.offset_x):
            msg = "planar offset_x must be finite"
            raise PulseError(msg)
        if self.propz != 0:
            msg = (
                "a planar launch is at the guide's entrance: propz must be 0, a "
                "free-space pre-propagation before the plates is not modelled"
            )
            raise PulseError(msg)
        if len(self.jones) != 1:
            msg = (
                "a planar beam is scalar: its component is the slab modes' own (y for "
                "TE, x for TM), so jones takes one entry, a phase"
            )
            raise PulseError(msg)

    def __repr__(self) -> str:
        """Return a one-line description: the two waists around the temporal pulse.

        Returns
        -------
        str
            For example ``PlanarPulse(waist_x=47 um, waist_y=2 mm, ...)``.
        """
        return line(
            "PlanarPulse",
            [
                ("waist_x", engineering(self.waist_x, "m")),
                ("waist_y", engineering(self.waist, "m")),
                ("", repr(self.temporal)),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "PlanarPulse: an astigmatic Gaussian beam into a planar guide", [
            ("waist in x", f"{engineering(self.waist_x, 'm')} (1/e field radius)"),
            ("offset in x", engineering(self.offset_x, "m")),
            ("waist in y", f"{engineering(self.waist, 'm')} (1/e field radius)"),
            ("order in y", self.order),
            ("temporal", str(self.temporal)),
        ]

    def build(
        self,
        grid: TimeGrid,
        space: RadialGrid | FreeGrid | PlanarGrid,
        *,
        medium: SpatialMedium,
        z: float = 0.0,
        side: FluxSide | None = None,
        frame: Frame | None = None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Build the launched ``(p, k_y)`` spectrum on a planar grid.

        The requested energy or peak power is the **incident** beam's physical
        axial flux in the fill medium, over the whole beam in x and the sampled y
        axis (SEP 34): the x integral of the continuous Gaussian's angular spectrum
        is taken by Gauss-Hermite quadrature of its admittance. The slab modes then
        carry the coupled share of that beam.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Field or envelope temporal grid.
        space : solprop.modes.slab.PlanarGrid
            The slab modes in x and the y axis; any other grid is refused.
        medium : solprop.spatial.SpatialMedium
            The fill medium at the entrance.
        z : float, optional
            Position of the entrance plane in m.
        side : {'before', 'after'} or None, optional
            Side of a declared index jump at ``z``; required there.
        frame : solprop.linear.Frame or None, optional
            Unused: the launch is at the entrance, with no pre-propagation.

        Returns
        -------
        numpy.ndarray
            Modal amplitudes in sqrt(W/m), raw-DFT transformed in y, times the
            temporal spectrum, shape ``(n_p*n_y, 1, n_omega)``, the mode index
            fastest, in the internal convention.

        Raises
        ------
        PulseError
            If the grid is not a planar one, the temporal pulse is not scalar, or
            the beam has no propagating content on the y axis.
        """
        del frame
        if not isinstance(space, PlanarGrid):
            msg = "a PlanarPulse launches into a PlanarGrid"
            raise PulseError(msg)
        limit = _GAUSS_EXPONENT_CLIP ** (1 / (2 * self.order))
        profile_y = np.exp(
            -(
                (np.minimum(np.abs(space.y), self.waist * limit) / self.waist)
                ** (2 * self.order)
            )
        )
        coupled = gaussian_projection(space.modes, self.waist_x, offset=self.offset_x)
        jones = complex(self.jones[0])
        phase = complex(fourier.to_internal(np.array([jones / abs(jones)]))[0])
        y_spectrum = np.fft.fft(profile_y)
        incident = _incident_admittance(grid, space, medium, z, side, self.waist_x)
        # The incident beam measured as y columns: the x integral of the unit-peak
        # Gaussian's |E|^2 is waist_x sqrt(pi/2), its admittance averaged in k_x.
        reference = _Plane(
            weights=space.dy / space.n_y * np.ones(space.n_y),
            admittance=incident,
            scale=0.5 * EPS0 * C * self.waist_x * math.sqrt(math.pi / 2),
        )
        columns = (y_spectrum[:, None] * coupled[None, :]).ravel()
        total = np.zeros((space.n_columns, 1, grid.n_omega), complex)
        for member in _members(self.temporal):
            temporal = member.build(grid)
            if temporal.shape != (1, 1, grid.n_omega):
                msg = "PlanarPulse requires a scalar single-column temporal pulse"
                raise PulseError(msg)
            beam = y_spectrum[:, None, None] * phase * temporal
            factor = _normalising_factor(member, temporal, beam, reference, grid)
            total += columns[:, None, None] * (phase * factor) * temporal
        if not np.all(np.isfinite(total)):
            msg = "planar launch produced a nonfinite field"
            raise PulseError(msg)
        return np.asarray(total, dtype=np.complex128)


@dataclass(frozen=True)
class _Plane:
    """Column weights and admittance measuring a launch's reference beam.

    Attributes
    ----------
    weights : numpy.ndarray
        Transverse Parseval weight per column.
    admittance : numpy.ndarray
        Dimensionless admittance per column and bin.
    scale : float
        Factor turning weighted ``|X|^2`` sums into joules.
    """

    weights: Float[np.ndarray, " n_modes"]
    admittance: Complex[np.ndarray, "n_modes n_omega"]
    scale: float


_HERMITE_ORDER = 64
"""Gauss-Hermite nodes averaging a Gaussian's admittance over its x wave numbers."""


def _incident_admittance(
    grid: TimeGrid,
    space: PlanarGrid,
    medium: SpatialMedium,
    z: float,
    side: FluxSide | None,
    waist_x: float,
) -> Complex[np.ndarray, "n_y n_omega"]:
    """Average the bulk admittance over an incident Gaussian's x wave numbers.

    The Gaussian ``exp(-x**2/waist_x**2)`` has the power spectrum
    ``exp(-k_x**2 waist_x**2 / 2)``, a normal density of standard deviation
    ``1/waist_x``; its admittance-weighted energy is the unweighted one times this
    average. The integrand is analytic away from cutoff, where its weight is
    negligible for any beam the grid resolves.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Temporal grid.
    space : solprop.modes.slab.PlanarGrid
        The y axis.
    medium : solprop.spatial.SpatialMedium
        The fill medium.
    z : float
        Position of the entrance plane in m.
    side : {'before', 'after'} or None
        Side of a declared index jump at ``z``.
    waist_x : float
        The 1/e field radius in x, m.

    Returns
    -------
    numpy.ndarray
        Complex dimensionless admittance, shape ``(n_y, n_omega)``.
    """
    frame = Frame(grid.omega0, subtract_beta0=False, beta0=0.0, beta1=0.0)
    material, _, _, _ = SpatialDispersion(medium, grid, frame).shared_values(
        plane_sample(medium, z, side)
    )
    nodes, weights = np.polynomial.hermite.hermgauss(_HERMITE_ORDER)
    square = material[None, :] - (space.k_y**2)[:, None]
    omega = grid.omega
    live = grid.active_mask[None, :] & (omega > 0)[None, :]
    average = np.zeros(square.shape, complex)
    for node, weight in zip(math.sqrt(2) * nodes / waist_x, weights, strict=True):
        longitudinal = square - node**2
        value = np.zeros(square.shape, complex)
        np.divide(
            C * np.sqrt(longitudinal),
            omega[None, :],
            out=value,
            where=live & (np.real(longitudinal) > 0),
        )
        average += weight * value
    return average / math.sqrt(math.pi)


# --- shared validation and machinery --------------------------------------------------


def _check_positive(name: str, value: float) -> None:
    """Reject a quantity that is not finite and strictly positive.

    Parameters
    ----------
    name : str
        Name of the quantity, used in the message.
    value : float
        Value to check.

    Raises
    ------
    PulseError
        If the value is not a finite positive number.
    """
    if not math.isfinite(value) or value <= 0.0:
        msg = f"{name} must be a finite positive number, got {value!r}"
        raise PulseError(msg)


def _scaling(energy: float | None, peak_power: float | None) -> tuple[float, _Quantity]:
    """Return the target and which quantity it is, rejecting an ambiguous request.

    Every pulse constructor shares this one definition and one message.

    Parameters
    ----------
    energy : float or None
        Requested pulse energy, J.
    peak_power : float or None
        Requested peak power, W.

    Returns
    -------
    tuple
        The target value and the name of the quantity it fixes.

    Raises
    ------
    PulseError
        If both or neither is given, or the given one is not finite and positive.
    """
    if energy is not None:
        if peak_power is not None:
            msg = (
                "only one of energy or peak_power can be given, got "
                f"energy={energy!r} and peak_power={peak_power!r}"
            )
            raise PulseError(msg)
        _check_positive("energy", energy)
        return energy, "energy"
    if peak_power is None:
        msg = "one of energy or peak_power must be given, got neither"
        raise PulseError(msg)
    _check_positive("peak_power", peak_power)
    return peak_power, "peak_power"


def _hard_masked(
    field: np.ndarray,
    live: np.ndarray,
    weights: np.ndarray,
    grid: TimeGrid,
) -> np.ndarray:
    """Zero a beam's bins without forward admittance, refusing real content there.

    The inactive band and zero frequency are the established hard masks. A
    nonpropagating column inside the active band is no forward power port either,
    and an energy-normalised launch requires it unoccupied (SEP 34 section 4.3):
    content beyond the rounding of the beam's field norm there is refused.

    Parameters
    ----------
    field : numpy.ndarray
        The beam spectrum, ``(n_modes, n_pol, n_omega)``.
    live : numpy.ndarray
        Bins with nonzero admittance, ``(n_modes, n_omega)``.
    weights : numpy.ndarray
        Transverse Parseval weights, ``(n_modes,)``.
    grid : solprop.grid.TimeGrid
        Temporal grid.

    Returns
    -------
    numpy.ndarray
        The masked beam.

    Raises
    ------
    PulseError
        If the nonpropagating share of the field norm exceeds its rounding budget.
    """
    band = grid.active_mask & (grid.omega > 0)
    norm = weights[:, None] * np.sum(np.abs(field) ** 2, axis=1)
    removed = float(np.sum(norm[band[None, :] & ~live]))
    total = float(np.sum(norm[:, band]))
    # The transforms that formed the beam carry a few rounding units of its norm
    # per bin they sum; content above that is physical.
    budget = 8 * field.shape[0] * grid.n_omega * np.finfo(float).eps
    if total > 0 and removed > budget * total:
        msg = (
            f"the beam puts {removed / total:.3e} of its field norm into columns "
            "that do not propagate at the launch plane, which carry no energy: "
            "widen the waist or refine the transverse grid"
        )
        raise PulseError(msg)
    return field * live[:, None, :]


def _members(pulse: Pulse) -> list[Pulse]:
    """Return the members of a pulse sum, flattened, or the pulse itself.

    Parameters
    ----------
    pulse : Pulse
        A temporal pulse or a :class:`PulseSum`.

    Returns
    -------
    list of Pulse
        Each member carries its own requested energy or peak power.
    """
    if isinstance(pulse, PulseSum):
        return [leaf for member in pulse.pulses for leaf in _members(member)]
    return [pulse]


def _target(
    pulse: Pulse, temporal: np.ndarray, grid: TimeGrid
) -> tuple[_Quantity, float]:
    """Return which quantity a temporal pulse fixes and its requested value.

    Parameters
    ----------
    pulse : Pulse
        One temporal pulse.
    temporal : numpy.ndarray
        Its built spectrum, shape ``(1, 1, n_omega)``.
    grid : solprop.grid.TimeGrid
        Temporal grid.

    Returns
    -------
    tuple
        ``("peak_power", W)`` for a pulse given a peak power, otherwise
        ``("energy", J)``: the energy its built spectrum carries.
    """
    peak = getattr(pulse, "peak_power", None)
    if isinstance(pulse, ShapedPulse) and peak is not None:
        return "peak_power", float(peak)
    energy = fourier.energy_omega(temporal[0, 0], grid.dt, kind=grid.kind)
    return "energy", float(energy)


def _normalising_factor(
    pulse: Pulse,
    temporal: np.ndarray,
    beam: np.ndarray,
    plane: _Plane,
    grid: TimeGrid,
) -> float:
    """Return the factor giving a beam its requested physical energy or peak power.

    Parameters
    ----------
    pulse : Pulse
        The temporal pulse fixing the request.
    temporal : numpy.ndarray
        Its built spectrum, shape ``(1, 1, n_omega)``.
    beam : numpy.ndarray
        The unnormalised beam's column spectra, ``(n_modes, n_pol, n_omega)``.
    plane : _Plane
        Its measure at the launch plane.
    grid : solprop.grid.TimeGrid
        Temporal grid.

    Returns
    -------
    float
        The amplitude factor.

    Raises
    ------
    PulseError
        If the beam carries no positive, finite forward flux.
    """
    quantity, target = _target(pulse, temporal, grid)
    if quantity == "energy":
        energy_weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
        density = np.sum(np.abs(beam) ** 2, axis=1)
        measured = plane.scale * float(
            np.sum(
                plane.weights[:, None]
                * energy_weights
                * np.real(plane.admittance)
                * density
            )
        )
    else:
        measured = float(
            np.max(
                power_trace(beam, plane.admittance, plane.weights, grid, plane.scale)
            )
        )
    if not (math.isfinite(measured) and math.isfinite(target)):
        msg = "the launch beam is nonfinite: its temporal pulse or profile overflowed"
        raise PulseError(msg)
    if measured <= 0:
        msg = (
            "the beam carries no positive forward flux at the launch plane: it is "
            "unresolved on this grid or lies outside the propagating band"
        )
        raise PulseError(msg)
    return math.sqrt(target / measured)


def _normalised(
    pulse: Pulse,
    temporal: np.ndarray,
    field: np.ndarray,
    flux: SpatialFlux,
    grid: TimeGrid,
) -> np.ndarray:
    """Scale a spatial beam to its requested physical energy or peak power.

    Parameters
    ----------
    pulse : Pulse
        The temporal pulse fixing the request.
    temporal : numpy.ndarray
        Its built spectrum, shape ``(1, 1, n_omega)``.
    field : numpy.ndarray
        The unnormalised beam spectrum, ``(n_modes, n_pol, n_omega)``.
    flux : solprop.spatial.SpatialFlux
        The launch plane's measure.
    grid : solprop.grid.TimeGrid
        Temporal grid.

    Returns
    -------
    numpy.ndarray
        The normalised beam.
    """
    plane = _Plane(
        weights=flux.space.spectral_weights,
        admittance=np.concatenate(
            [flux.admittance(columns) for columns in flux.chunks()]
        ),
        scale=flux.energy_scale,
    )
    return field * _normalising_factor(pulse, temporal, field, plane, grid)


def _frozen_phase(phase: Sequence[float] | np.ndarray) -> Float[np.ndarray, " n_deg"]:
    """Return the Taylor coefficients as a validated read-only array.

    Parameters
    ----------
    phase : array_like
        Spectral phase Taylor coefficients about the pulse's centre frequency: radians
        for the coefficient of order 0 and ``s**n`` for order ``n`` thereafter.

    Returns
    -------
    numpy.ndarray
        A read-only copy, shape ``(n_deg,)``.

    Raises
    ------
    PulseError
        If any coefficient is not finite.
    """
    coefficients = np.array(phase, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(coefficients)):
        msg = f"every spectral phase coefficient must be finite, got {phase!r}"
        raise PulseError(msg)
    coefficients.setflags(write=False)
    return coefficients


def _check_delay(grid: TimeGrid, t0: float) -> None:
    """Reject a delay that would move the pulse out of the time window.

    The delay is applied as a spectral phase, so a value past the half-window wraps the
    pulse round to the other end rather than moving it out of the way.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Grid the pulse is built on.
    t0 : float
        Requested delay, s.

    Raises
    ------
    PulseError
        If ``t0`` is not finite, or reaches the edge of the window.
    """
    half = 0.5 * grid.n_t * grid.dt
    if not math.isfinite(t0) or abs(t0) >= half:
        msg = (
            f"t0 must lie strictly inside the time window, |t0| < {half:.4g} s, "
            f"got {t0!r}"
        )
        raise PulseError(msg)


def _phase_factor(
    grid: TimeGrid,
    phase: Float[np.ndarray, " n_deg"],
    t0: float,
    omega_ref: float,
) -> Complex[np.ndarray, " n_omega"]:
    """Return the internal-convention phasor for a delay and a Taylor spectral phase.

    The optics-convention phase is ``phi(omega) = omega t0 + sum_n phi_n (omega -
    omega_ref)**n / n!``, so that the group delay ``dphi/domega`` is ``t0`` at
    ``omega_ref`` and a positive second coefficient makes the higher frequencies arrive
    later. It is converted once, here, with :func:`solprop.fourier.to_internal`.

    The absolute ``omega`` is used on both grids, so a
    delay moves the whole field rather than only its envelope: the carrier-envelope
    phase slips by ``omega0 t0``, and the field and envelope representations stay in
    step. A Taylor phase and a pure delay both commute with the half-window origin
    shift, so they are applied to the stored spectrum directly (conventions.md §2).

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        Grid supplying the frequency axis.
    phase : numpy.ndarray
        Taylor coefficients, shape ``(n_deg,)``, in ``s**n`` for order ``n``.
    t0 : float
        Delay, s, positive for later arrival.
    omega_ref : float
        Frequency the Taylor expansion is about, rad/s.

    Returns
    -------
    numpy.ndarray
        Unit-modulus factor, shape ``(n_omega,)``, to multiply a stored spectrum by.
    """
    optics_phase = taylor_phase(grid.omega, phase, omega_ref, t0)
    return fourier.to_internal(np.exp(1j * optics_phase))


def _analytic(
    spectrum: Complex[np.ndarray, " n_omega"], grid: TimeGrid
) -> Complex[np.ndarray, " n_t"]:
    """Return the analytic signal of a stored spectrum, whose modulus squared is power.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Stored spectrum, shape ``(n_omega,)``.
    grid : solprop.grid.TimeGrid
        Grid the spectrum belongs to.

    Returns
    -------
    numpy.ndarray
        Analytic signal, shape ``(n_t,)``, in the square root of watts.
    """
    samples = fourier.inverse(spectrum, kind=grid.kind)
    if grid.kind == "envelope":
        return samples.astype(complex)
    return fourier.analytic_signal(np.real(samples))


def _realised(
    spectrum: Complex[np.ndarray, " n_omega"], grid: TimeGrid, quantity: _Quantity
) -> float:
    """Return the value a spectrum already has of the quantity being scaled.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Stored spectrum, shape ``(n_omega,)``.
    grid : solprop.grid.TimeGrid
        Grid the spectrum belongs to.
    quantity : {'energy', 'peak_power', 'average_power'}
        Which quantity to measure.

    Returns
    -------
    float
        The energy in J, or the peak or mean power in W. The mean is taken over the flat
        part of the temporal window, where the collar is exactly one.
    """
    if quantity == "energy":
        return float(fourier.energy_omega(spectrum, grid.dt, kind=grid.kind))
    power = np.abs(_analytic(spectrum, grid)) ** 2
    if quantity == "peak_power":
        return float(np.max(power))
    return float(np.mean(power[grid.t_window == 1.0]))


def _rescale(
    spectrum: Complex[np.ndarray, " n_omega"],
    grid: TimeGrid,
    target: float,
    quantity: _Quantity,
) -> Complex[np.ndarray, " n_omega"]:
    """Scale a spectrum so that the named quantity takes the requested value.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Stored spectrum, shape ``(n_omega,)``.
    grid : solprop.grid.TimeGrid
        Grid the spectrum belongs to.
    target : float
        Requested value, J for an energy and W for a power.
    quantity : {'energy', 'peak_power', 'average_power'}
        Which quantity ``target`` fixes.

    Returns
    -------
    numpy.ndarray
        The scaled spectrum, shape ``(n_omega,)``.

    Raises
    ------
    PulseError
        If the spectrum carries nothing to scale, so no factor exists.
    """
    realised = _realised(spectrum, grid, quantity)
    if realised <= 0.0:
        msg = (
            f"the pulse has no {quantity.replace('_', ' ')} on this grid, so it cannot "
            "be scaled; check that its shape and centre wavelength fall inside the "
            "grid's window"
        )
        raise PulseError(msg)
    return spectrum * math.sqrt(target / realised)


def _canonical(
    spectrum: Complex[np.ndarray, " n_omega"],
) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
    """Return a single-mode spectrum in the canonical kernel layout.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Stored spectrum, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The same spectrum, shape ``(1, 1, n_omega)``, complex128 and C-contiguous.
    """
    return np.ascontiguousarray(spectrum, dtype=np.complex128)[np.newaxis, np.newaxis]


# --- pulses with a power envelope in time ---------------------------------------------


@dataclass(frozen=True, eq=False, kw_only=True)
class ShapedPulse(Displayed):
    """Common parameters and build path of a pulse defined by its power envelope.

    :class:`GaussianPulse`, :class:`SechPulse` and :class:`CustomPulse` derive from this
    class: it holds the parameters they share and the whole of :meth:`build` (the
    carrier, the spectral phase, the exact scaling and the canonical layout), so that
    the three differ only in one line of arithmetic. Subclass it and supply
    :meth:`intensity` to add a shape of your own; it is abstract here.

    Attributes
    ----------
    wavelength : float
        Centre wavelength of the pulse, m. Also the frequency the Taylor spectral phase
        is expanded about, which need not be the grid's reference wavelength.
    energy : float or None
        Pulse energy, J. Exactly one of ``energy`` and ``peak_power`` is given.
    peak_power : float or None
        Peak power, W, realised *after* the spectral phase is applied.
    phase : array_like
        Optics-convention spectral phase as Taylor coefficients about ``wavelength``:
        ``[CEP (rad), group delay (s), GDD (s**2), TOD (s**3), ...]``. Empty by default.
    t0 : float
        Delay, s, positive for later arrival. Must lie strictly inside the window.
    """

    wavelength: float
    energy: float | None = None
    peak_power: float | None = None
    phase: Sequence[float] | np.ndarray = ()
    t0: float = 0.0

    def __post_init__(self) -> None:
        """Validate the shared parameters and freeze the Taylor coefficients.

        Raises
        ------
        PulseError
            If the wavelength is not positive, the scaling request is ambiguous, or a
            phase coefficient is not finite.
        """
        _check_positive("wavelength", self.wavelength)
        _scaling(self.energy, self.peak_power)
        object.__setattr__(self, "phase", _frozen_phase(self.phase))

    def _shape_fields(self) -> list[tuple[str, object]]:
        """Return the fields a subclass adds, for printing.

        Returns
        -------
        list of tuple
            ``(key, value)`` pairs describing the shape: its duration, its order.
        """
        return []

    def _scale(self) -> tuple[str, str]:
        """Return which of energy and peak power scales the pulse, for printing.

        Returns
        -------
        tuple of str
            The key and the formatted value.
        """
        if self.energy is not None:
            return "energy", engineering(self.energy, "J")
        return "peak_power", engineering(self.peak_power, "W")

    def __repr__(self) -> str:
        """Return a one-line description with the phase summarised, never dumped.

        Returns
        -------
        str
            For example ``GaussianPulse(wavelength=800 nm, fwhm=30 fs, energy=1 uJ)``.
        """
        fields: list[tuple[str, object]] = [
            ("wavelength", engineering(self.wavelength, "m")),
            *self._shape_fields(),
            self._scale(),
        ]
        if len(self.phase):
            fields.append(("phase", f"[{phase_text(self.phase)}]"))
        if self.t0:
            fields.append(("t0", engineering(self.t0, "s")))
        return line(type(self).__name__, fields)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return type(self).__name__, [
            ("wavelength", engineering(self.wavelength, "m")),
            *self._shape_fields(),
            self._scale(),
            ("spectral phase", phase_text(self.phase)),
            ("delay", engineering(self.t0, "s")),
        ]

    def intensity(self, t: Float[np.ndarray, " n_t"]) -> Float[np.ndarray, " n_t"]:
        """Return the power envelope of the pulse, in arbitrary units.

        The normalisation is irrelevant: the spectrum is rescaled to the requested
        energy or peak power afterwards.

        Parameters
        ----------
        t : numpy.ndarray
            Time samples, s, shape ``(n_t,)``.

        Returns
        -------
        numpy.ndarray
            Non-negative power envelope, shape ``(n_t,)``.

        Raises
        ------
        NotImplementedError
            Always; each shaped pulse supplies its own envelope.
        """
        raise NotImplementedError

    def build(self, grid: TimeGrid) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the initial spectrum of this pulse on the given grid.

        On a field grid the samples are ``sqrt(I(t)) cos(omega_p t)``, whose analytic
        signal is the pulse; on an envelope grid they are
        ``sqrt(I(t)) exp(i (omega_p - omega0) t)``, which is that analytic signal with
        the grid's carrier removed (conventions.md §4). The spectral phase is applied to
        the stored spectrum and the scaling comes last.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Grid to build on.

        Returns
        -------
        numpy.ndarray
            Stored spectrum, shape ``(1, 1, n_omega)``, in the square root of watts.

        Raises
        ------
        PulseError
            If the delay leaves the window, the envelope is not finite and non-negative,
            or the pulse carries nothing to scale on this grid.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop import fourier
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.pulses import GaussianPulse
        >>> grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 2e-12)
        >>> spectrum = GaussianPulse(
        ...     wavelength=800e-9, fwhm=30e-15, energy=1e-6
        ... ).build(grid)
        >>> energy = fourier.energy_omega(spectrum[0, 0], grid.dt, kind=grid.kind)
        >>> bool(np.isclose(energy, 1e-6, rtol=1e-12))
        True
        """
        _check_delay(grid, self.t0)
        target, quantity = _scaling(self.energy, self.peak_power)
        omega_p = 2.0 * np.pi * C / self.wavelength
        amplitude = np.sqrt(_power_envelope(self, grid))
        if grid.kind == "field":
            samples = amplitude * np.cos(omega_p * grid.t)
        else:
            samples = amplitude * np.exp(1j * (omega_p - grid.omega0) * grid.t)
        spectrum = fourier.forward(samples, kind=grid.kind)
        phase = np.asarray(self.phase)
        spectrum = spectrum * _phase_factor(grid, phase, self.t0, omega_p)
        return _canonical(_rescale(spectrum, grid, target, quantity))


@dataclass(frozen=True, eq=False, repr=False)
class PulseSum(Displayed):
    """Several pulses launched together: the input is the sum of their fields.

    Two-colour and multi-colour inputs -- a pump and a seed for four-wave mixing, a
    strong pump and a weak probe for cross-phase modulation, a pulse and its delayed
    replica -- are the sum of independently specified pulses. Each member keeps its own
    centre wavelength, duration, energy or peak power, spectral phase and delay, and is
    built and scaled on the grid exactly as it would be alone; the spectra are then
    added. The energies of well-separated members therefore add, and members that
    overlap in time and frequency interfere, as the fields do.

    Attributes
    ----------
    pulses : tuple of Pulse
        The members, at least one; any object with a ``build(grid)`` method whose
        spectra share a shape, typically :class:`GaussianPulse`, :class:`SechPulse`,
        :class:`CustomPulse` or :class:`DataPulse`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop import fourier
    >>> from solprop.grid import FieldGrid
    >>> from solprop.pulses import GaussianPulse, PulseSum
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (150e-9, 3e-6), 1e-12)
    >>> pump = GaussianPulse(wavelength=267e-9, fwhm=35e-15, energy=95e-6)
    >>> seed = GaussianPulse(wavelength=800e-9, fwhm=35e-15, energy=45e-6)
    >>> spectrum = PulseSum((pump, seed)).build(grid)
    >>> energy = fourier.energy_omega(spectrum[0, 0], grid.dt, kind=grid.kind)
    >>> bool(np.isclose(energy, 140e-6, rtol=1e-6))
    True
    """

    pulses: tuple[Pulse, ...]

    def __post_init__(self) -> None:
        """Freeze the members and check that each can be built.

        Raises
        ------
        PulseError
            If there are no members, or a member has no ``build`` method.
        """
        members = tuple(self.pulses)
        if not members:
            msg = "a PulseSum needs at least one pulse"
            raise PulseError(msg)
        for index, member in enumerate(members):
            if not callable(getattr(member, "build", None)):
                msg = (
                    f"member {index} of the PulseSum is a {type(member).__name__}, "
                    "which has no build(grid) method: give solprop.pulses objects"
                )
                raise PulseError(msg)
        object.__setattr__(self, "pulses", members)

    def __repr__(self) -> str:
        """Return a one-line description listing the members.

        Returns
        -------
        str
            For example ``PulseSum(GaussianPulse(...), GaussianPulse(...))``.
        """
        return f"PulseSum({', '.join(repr(member) for member in self.pulses)})"

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and one row per member.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "PulseSum", [
            (f"pulse {index}", repr(member)) for index, member in enumerate(self.pulses)
        ]

    def build(self, grid: TimeGrid) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the sum of the members' spectra on the given grid.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Grid to build on; every member is built on it independently.

        Returns
        -------
        numpy.ndarray
            Stored spectrum, the members' common shape, in the square root of watts.

        Raises
        ------
        PulseError
            If the members build to spectra of different shapes.
        """
        spectra = [np.asarray(member.build(grid)) for member in self.pulses]
        shapes = {spectrum.shape for spectrum in spectra}
        if len(shapes) != 1:
            msg = (
                f"the members of a PulseSum build to spectra of different shapes "
                f"{sorted(shapes)}; combine pulses of one kind"
            )
            raise PulseError(msg)
        return np.sum(spectra, axis=0).astype(np.complex128)


def _power_envelope(pulse: ShapedPulse, grid: TimeGrid) -> Float[np.ndarray, " n_t"]:
    """Evaluate a pulse's envelope on the grid and check that it is usable.

    Parameters
    ----------
    pulse : ShapedPulse
        Pulse whose :meth:`ShapedPulse.intensity` is evaluated.
    grid : solprop.grid.TimeGrid
        Grid supplying the time axis.

    Returns
    -------
    numpy.ndarray
        The power envelope, shape ``(n_t,)``.

    Raises
    ------
    PulseError
        If the envelope has the wrong length, is not finite, is negative anywhere, or is
        zero at every sample.
    """
    values = np.asarray(pulse.intensity(grid.t), dtype=np.float64)
    if values.shape != grid.t.shape:
        msg = (
            f"the pulse envelope must have one value per time sample, shape "
            f"{grid.t.shape}, got {values.shape}"
        )
        raise PulseError(msg)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        msg = "the pulse envelope must be finite and non-negative at every sample"
        raise PulseError(msg)
    if not np.any(values > 0.0):
        msg = (
            "the pulse envelope is zero at every sample of this grid; check the pulse "
            "duration against the time window"
        )
        raise PulseError(msg)
    return values


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class GaussianPulse(ShapedPulse):
    """A (super-)Gaussian pulse of given intensity FWHM.

    The envelope is ``exp(-(t/sigma)**(2m) / 2)`` with
    ``sigma = fwhm / (2 (2 ln 2)**(1/2m))``, which realises the requested FWHM for every
    order ``m``.

    Attributes
    ----------
    wavelength : float
        Centre wavelength of the pulse, m, and the reference of the Taylor phase.
    fwhm : float
        Full width at half maximum of the power envelope, s.
    m : int
        Super-Gaussian order, at least 1. The exponent of the envelope is ``2m``, so
        ``m = 1`` is an ordinary Gaussian and large ``m`` approaches a flat top.
    energy : float or None
        Pulse energy, J. Exactly one of ``energy`` and ``peak_power`` is given.
    peak_power : float or None
        Peak power, W, realised after the spectral phase.
    phase : array_like
        Optics-convention Taylor coefficients about ``wavelength``: radians at order 0
        and ``s**n`` at order ``n`` thereafter.
    t0 : float
        Delay, s, positive for later arrival.

    Raises
    ------
    PulseError
        If the duration or the order is invalid, or the scaling request is ambiguous.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.pulses import GaussianPulse
    >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    >>> float(pulse.intensity(np.array([0.0, 15e-15]))[1])
    0.5
    """

    fwhm: float
    m: int = 1

    def __post_init__(self) -> None:
        """Validate the duration and the super-Gaussian order.

        Raises
        ------
        PulseError
            If ``fwhm`` is not positive or ``m`` is less than one.
        """
        super().__post_init__()
        _check_positive("fwhm", self.fwhm)
        if self.m < 1:
            msg = f"m must be a super-Gaussian order of at least 1, got {self.m!r}"
            raise PulseError(msg)

    def _shape_fields(self) -> list[tuple[str, object]]:
        """Return the duration, and the order when it is not the plain Gaussian's.

        Returns
        -------
        list of tuple
            ``(key, value)`` pairs.
        """
        fields: list[tuple[str, object]] = [("fwhm", engineering(self.fwhm, "s"))]
        if self.m != 1:
            fields.append(("m", self.m))
        return fields

    def intensity(self, t: Float[np.ndarray, " n_t"]) -> Float[np.ndarray, " n_t"]:
        """Return the (super-)Gaussian power envelope, normalised to a peak of one.

        Parameters
        ----------
        t : numpy.ndarray
            Time samples, s, shape ``(n_t,)``.

        Returns
        -------
        numpy.ndarray
            Envelope values in ``[0, 1]``, shape ``(n_t,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.pulses import GaussianPulse
        >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=2.0, energy=1.0, m=3)
        >>> float(round(pulse.intensity(np.array([1.0]))[0], 12))
        0.5
        """
        order = 2 * self.m
        sigma = self.fwhm / (2.0 * (2.0 * math.log(2.0)) ** (1.0 / order))
        limit = (2.0 * _GAUSS_EXPONENT_CLIP) ** (1.0 / order)
        scaled = np.minimum(np.abs(np.asarray(t) / sigma), limit)
        return np.exp(-0.5 * scaled**order)


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class SechPulse(ShapedPulse):
    """A ``sech**2`` pulse, specified by its FWHM or by its natural width.

    Attributes
    ----------
    wavelength : float
        Centre wavelength of the pulse, m, and the reference of the Taylor phase.
    fwhm : float or None
        Full width at half maximum of the power envelope, s. Exactly one of ``fwhm`` and
        ``width`` is given.
    width : float or None
        Natural width of ``sech**2(t/width)``, s, which is
        ``fwhm / (2 ln(1 + sqrt(2)))``.
    energy : float or None
        Pulse energy, J. Exactly one of ``energy`` and ``peak_power`` is given.
    peak_power : float or None
        Peak power, W, realised after the spectral phase.
    phase : array_like
        Optics-convention Taylor coefficients about ``wavelength``: radians at order 0
        and ``s**n`` at order ``n`` thereafter.
    t0 : float
        Delay, s, positive for later arrival.

    Raises
    ------
    PulseError
        If both or neither duration is given, or the scaling request is ambiguous.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.pulses import SechPulse
    >>> pulse = SechPulse(wavelength=1030e-9, fwhm=100e-15, energy=10e-6)
    >>> float(round(pulse.intensity(np.array([50e-15]))[0], 12))
    0.5
    """

    fwhm: float | None = None
    width: float | None = None

    def __post_init__(self) -> None:
        """Validate that exactly one duration is given and is positive.

        Raises
        ------
        PulseError
            If both or neither of ``fwhm`` and ``width`` is given, or one is not
            positive.
        """
        super().__post_init__()
        _sech_width(self.fwhm, self.width)

    def _shape_fields(self) -> list[tuple[str, object]]:
        """Return whichever of the two durations was given.

        Returns
        -------
        list of tuple
            ``(key, value)`` pairs.
        """
        if self.fwhm is not None:
            return [("fwhm", engineering(self.fwhm, "s"))]
        return [("width", engineering(self.width, "s"))]

    def intensity(self, t: Float[np.ndarray, " n_t"]) -> Float[np.ndarray, " n_t"]:
        """Return the ``sech**2`` power envelope, normalised to a peak of one.

        Parameters
        ----------
        t : numpy.ndarray
            Time samples, s, shape ``(n_t,)``.

        Returns
        -------
        numpy.ndarray
            Envelope values in ``(0, 1]``, shape ``(n_t,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.pulses import SechPulse
        >>> pulse = SechPulse(wavelength=1030e-9, width=1.0, energy=1.0)
        >>> float(pulse.intensity(np.array([0.0]))[0])
        1.0
        """
        width = _sech_width(self.fwhm, self.width)
        scaled = np.minimum(np.abs(np.asarray(t) / width), _SECH_CLIP)
        return 1.0 / np.cosh(scaled) ** 2


def _sech_width(fwhm: float | None, width: float | None) -> float:
    """Return the natural width of a ``sech**2`` pulse from whichever duration is given.

    Parameters
    ----------
    fwhm : float or None
        Intensity full width at half maximum, s.
    width : float or None
        Natural width, s.

    Returns
    -------
    float
        The natural width, s.

    Raises
    ------
    PulseError
        If both or neither is given, or the given one is not finite and positive.
    """
    if width is not None:
        if fwhm is not None:
            msg = (
                "only one of fwhm or width can be given, got "
                f"fwhm={fwhm!r} and width={width!r}"
            )
            raise PulseError(msg)
        _check_positive("width", width)
        return width
    if fwhm is None:
        msg = "one of fwhm or width must be given, got neither"
        raise PulseError(msg)
    _check_positive("fwhm", fwhm)
    return fwhm * SECH_FWHM_TO_WIDTH


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class CustomPulse(ShapedPulse):
    """A pulse whose power envelope is an arbitrary callable of time.

    Attributes
    ----------
    wavelength : float
        Centre wavelength of the pulse, m, and the reference of the Taylor phase.
    shape_fn : callable
        ``I(t)`` returning the power envelope for an array of times in seconds. It must
        be finite and non-negative on the grid; its normalisation is irrelevant.
    energy : float or None
        Pulse energy, J. Exactly one of ``energy`` and ``peak_power`` is given.
    peak_power : float or None
        Peak power, W, realised after the spectral phase.
    phase : array_like
        Optics-convention Taylor coefficients about ``wavelength``: radians at order 0
        and ``s**n`` at order ``n`` thereafter.
    t0 : float
        Delay, s, positive for later arrival.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.pulses import CustomPulse
    >>> triangle = CustomPulse(
    ...     wavelength=800e-9,
    ...     shape_fn=lambda t: np.clip(1.0 - np.abs(t) / 20e-15, 0.0, None),
    ...     energy=1e-6,
    ... )
    >>> float(triangle.intensity(np.array([10e-15]))[0])
    0.5
    """

    shape_fn: Callable[[Float[np.ndarray, " n_t"]], Float[np.ndarray, " n_t"]]

    def _shape_fields(self) -> list[tuple[str, object]]:
        """Return the name of the shape function.

        Returns
        -------
        list of tuple
            ``(key, value)`` pairs.
        """
        return [("shape_fn", getattr(self.shape_fn, "__name__", "callable"))]

    def intensity(self, t: Float[np.ndarray, " n_t"]) -> Float[np.ndarray, " n_t"]:
        """Return the caller's power envelope evaluated on the time samples.

        Parameters
        ----------
        t : numpy.ndarray
            Time samples, s, shape ``(n_t,)``.

        Returns
        -------
        numpy.ndarray
            The value of ``shape_fn(t)``, shape ``(n_t,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.pulses import CustomPulse
        >>> flat = CustomPulse(
        ...     wavelength=800e-9, shape_fn=np.ones_like, energy=1e-6
        ... )
        >>> flat.intensity(np.zeros(3)).tolist()
        [1.0, 1.0, 1.0]
        """
        return np.asarray(self.shape_fn(t), dtype=np.float64)


# --- pulses from measured data --------------------------------------------------------


@dataclass(frozen=True, eq=False, kw_only=True)
class DataPulse(Displayed):
    """A pulse from a measured complex spectrum, interpolated onto the grid.

    The stored form is the spectrum because that is what a measurement gives: a spectral
    power density and a retrieved phase. The phase is **absolute** and referenced to
    ``t = 0`` in the optics convention, so it fixes where in the window the pulse sits;
    it therefore goes through :func:`solprop.fourier.spectrum_to_internal`, which
    carries the half-window origin shift as well as the conjugation.

    Magnitude and unwrapped phase are interpolated separately with a cubic B-spline,
    never the complex spectrum itself, whose real and imaginary parts oscillate. Outside
    the data's own frequency range the spectrum is zero, and the grid's spectral collar
    is applied so that nothing is left sitting on the window edge.

    Attributes
    ----------
    omega : numpy.ndarray
        Angular frequency axis of the data, rad/s, shape ``(n_data,)``, strictly
        increasing and positive. It need not be uniform.
    spectrum : numpy.ndarray
        Measured complex spectrum on ``omega``, shape ``(n_data,)``, optics convention,
        arbitrary units.
    energy : float
        Pulse energy to scale to, J.
    phase : array_like
        Further optics-convention Taylor coefficients, applied on top of the phase
        already in the data, about ``wavelength``; radians at order 0 and ``s**n``
        thereafter.
    t0 : float
        Further delay, s, on top of the arrival time the data's own phase implies.
    wavelength : float or None
        Reference wavelength of the Taylor phase, m. Defaults to the spectral centre of
        mass of the data.

    Raises
    ------
    PulseError
        If the two arrays disagree in length, are too short to interpolate, are not
        finite, or ``omega`` is not strictly increasing and positive.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.pulses import DataPulse
    >>> omega = np.linspace(1.5e15, 3.5e15, 256)
    >>> intensity = np.exp(-(((omega - 2.35e15) / 1e14) ** 2))
    >>> pulse = DataPulse.from_spectral_density(
    ...     omega, intensity, np.zeros_like(omega), energy=1e-6
    ... )
    >>> float(round(pulse.reference_wavelength * 1e9))
    802.0
    """

    omega: Float[np.ndarray, " n_data"]
    spectrum: Complex[np.ndarray, " n_data_spectrum"]
    energy: float
    phase: Sequence[float] | np.ndarray = ()
    t0: float = 0.0
    wavelength: float | None = None

    def __post_init__(self) -> None:
        """Validate the tabulated data and adopt read-only copies of both arrays.

        Raises
        ------
        PulseError
            If the arrays disagree in length, are shorter than a cubic spline needs, are
            not finite, or ``omega`` is not strictly increasing and positive.
        """
        omega = np.array(self.omega, dtype=np.float64).reshape(-1)
        spectrum = np.array(self.spectrum, dtype=np.complex128).reshape(-1)
        if omega.size != spectrum.size:
            msg = (
                f"omega and spectrum must have the same length, got {omega.size} and "
                f"{spectrum.size}"
            )
            raise PulseError(msg)
        if omega.size < _MIN_DATA_POINTS:
            msg = (
                f"tabulated data needs at least {_MIN_DATA_POINTS} points for a cubic "
                f"interpolation, got {omega.size}"
            )
            raise PulseError(msg)
        if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(spectrum)):
            msg = "omega and spectrum must be finite everywhere"
            raise PulseError(msg)
        if omega[0] <= 0.0 or np.any(np.diff(omega) <= 0.0):
            msg = "omega must be strictly increasing and positive, in rad/s"
            raise PulseError(msg)
        _check_positive("energy", self.energy)
        if self.wavelength is not None:
            _check_positive("wavelength", self.wavelength)
        omega.setflags(write=False)
        spectrum.setflags(write=False)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "spectrum", spectrum)
        object.__setattr__(self, "phase", _frozen_phase(self.phase))

    @property
    def reference_wavelength(self) -> float:
        """Return the wavelength the Taylor phase is expanded about, m.

        Returns
        -------
        float
            ``wavelength`` if it was given, otherwise the wavelength of the spectral
            centre of mass of the data.
        """
        return 2.0 * np.pi * C / self._reference_frequency()

    def _reference_frequency(self) -> float:
        """Return the frequency the Taylor phase is expanded about, rad/s.

        Returns
        -------
        float
            The frequency of ``wavelength``, else the weighted mean of ``omega``.
        """
        if self.wavelength is not None:
            return 2.0 * np.pi * C / self.wavelength
        weight = np.abs(self.spectrum) ** 2
        return float(np.sum(self.omega * weight) / np.sum(weight))

    def __repr__(self) -> str:
        """Return a one-line description with the data summarised, never dumped.

        Returns
        -------
        str
            For example ``DataPulse(<512 values, 600 nm to 1 um>, energy=1 uJ)``.
        """
        return line(
            "DataPulse",
            [("", self._data_text()), ("energy", engineering(self.energy, "J"))],
        )

    def _data_text(self) -> str:
        """Return the extent of the measured spectrum, for printing.

        Returns
        -------
        str
            The number of samples and the wavelengths they span.
        """
        wavelengths = 2.0 * np.pi * C / np.asarray(self.omega, dtype=float)
        return summarise(wavelengths, "m")

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        reference = (
            "the data's centre"
            if self.wavelength is None
            else engineering(self.wavelength, "m")
        )
        return "DataPulse: a measured complex spectrum", [
            ("data", self._data_text()),
            ("energy", engineering(self.energy, "J")),
            ("further phase", f"{phase_text(self.phase)} about {reference}"),
            ("further delay", engineering(self.t0, "s")),
        ]

    def build(self, grid: TimeGrid) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the initial spectrum of the tabulated pulse on the given grid.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Grid to build on.

        Returns
        -------
        numpy.ndarray
            Stored spectrum, shape ``(1, 1, n_omega)``, in the square root of watts.

        Raises
        ------
        PulseError
            If the delay leaves the window, or the data does not reach the grid's active
            band at all.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.grid import FieldGrid
        >>> from solprop.pulses import DataPulse
        >>> omega = np.linspace(4e14, 1.1e16, 4096)  # covers the grid's active band
        >>> intensity = np.exp(-(((omega - 2.35e15) / 1e14) ** 2))
        >>> pulse = DataPulse.from_spectral_density(
        ...     omega, intensity, np.zeros_like(omega), energy=1e-6
        ... )
        >>> grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 400e-15)
        >>> pulse.build(grid).shape[:2]
        (1, 1)
        """
        _check_delay(grid, self.t0)
        inside = (grid.omega >= self.omega[0]) & (grid.omega <= self.omega[-1])
        if not np.any(inside & grid.active_mask):
            msg = (
                f"the data covers {self.omega[0]:.4g} to {self.omega[-1]:.4g} rad/s, "
                "which does not reach the grid's active band"
            )
            raise PulseError(msg)
        active = grid.omega[grid.active_mask]
        if active.min() < self.omega[0] or active.max() > self.omega[-1]:
            logger.warning(
                "the tabulated spectrum covers %.4g to %.4g rad/s but the grid's "
                "active band runs from %.4g to %.4g rad/s; the input is zero outside "
                "the data",
                self.omega[0],
                self.omega[-1],
                active.min(),
                active.max(),
            )
        magnitude = make_interp_spline(self.omega, np.abs(self.spectrum), k=3)
        angle = make_interp_spline(self.omega, np.unwrap(np.angle(self.spectrum)), k=3)
        optics = np.zeros(grid.n_omega, dtype=np.complex128)
        wanted = grid.omega[inside]
        optics[inside] = np.clip(magnitude(wanted), 0.0, None) * np.exp(
            1j * angle(wanted)
        )
        optics = optics * grid.omega_window
        spectrum = fourier.spectrum_to_internal(optics, kind=grid.kind)
        phase = np.asarray(self.phase)
        spectrum = spectrum * _phase_factor(
            grid, phase, self.t0, self._reference_frequency()
        )
        return _canonical(_rescale(spectrum, grid, self.energy, "energy"))

    @classmethod
    def from_spectral_density(  # noqa: PLR0913 - the data triple plus the shared parameters
        cls,
        omega: Float[np.ndarray, " n_data"],
        intensity: Float[np.ndarray, " n_data_intensity"],
        spectral_phase: Float[np.ndarray, " n_data_phase"],
        *,
        energy: float,
        phase: Sequence[float] | np.ndarray = (),
        t0: float = 0.0,
        wavelength: float | None = None,
        # The return annotation stays quoted: beartype resolves it while the class body
        # is still executing, when the bare class name is not yet bound.
    ) -> "DataPulse":  # noqa: UP037
        """Build a pulse from a spectral power density and an unwrapped spectral phase.

        This is the form a spectrometer and a phase retrieval give. Negative densities,
        which fits and baseline subtraction can produce, are clipped to zero.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequency axis, rad/s, shape ``(n_data,)``, strictly increasing.
        intensity : numpy.ndarray
            Spectral power density on ``omega``, arbitrary units, shape ``(n_data,)``.
        spectral_phase : numpy.ndarray
            Unwrapped optics-convention spectral phase on ``omega``, rad, shape
            ``(n_data,)``.
        energy : float
            Pulse energy to scale to, J.
        phase : array_like, optional
            Further Taylor coefficients about ``wavelength``, ``s**n`` at order ``n``.
        t0 : float, optional
            Further delay, s.
        wavelength : float or None, optional
            Reference wavelength of the Taylor phase, m.

        Returns
        -------
        DataPulse
            The pulse, with ``spectrum = sqrt(intensity) exp(i spectral_phase)``.

        Raises
        ------
        PulseError
            If the three arrays disagree in length, or the data fails validation.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.pulses import DataPulse
        >>> omega = np.linspace(2.0e15, 2.7e15, 64)
        >>> pulse = DataPulse.from_spectral_density(
        ...     omega, np.ones_like(omega), np.zeros_like(omega), energy=1e-6
        ... )
        >>> pulse.spectrum.shape
        (64,)
        """
        density = np.asarray(intensity, dtype=np.float64).reshape(-1)
        angle = np.asarray(spectral_phase, dtype=np.float64).reshape(-1)
        if density.size != angle.size:
            msg = (
                f"intensity and spectral_phase must have the same length, got "
                f"{density.size} and {angle.size}"
            )
            raise PulseError(msg)
        spectrum = np.sqrt(np.clip(density, 0.0, None)) * np.exp(1j * angle)
        return cls(
            omega=np.asarray(omega, dtype=np.float64).reshape(-1),
            spectrum=spectrum,
            energy=energy,
            phase=phase,
            t0=t0,
            wavelength=wavelength,
        )

    @classmethod
    def from_time(
        cls,
        t: Float[np.ndarray, " n_data"],
        field: Inexact[np.ndarray, " n_data_field"],
        *,
        energy: float,
        phase: Sequence[float] | np.ndarray = (),
        t0: float = 0.0,
        wavelength: float | None = None,
        # The return annotation stays quoted: beartype resolves it while the class body
        # is still executing, when the bare class name is not yet bound.
    ) -> "DataPulse":  # noqa: UP037
        """Build a pulse from measured time-domain samples on a uniform axis.

        The optics-convention spectrum referenced to ``t = 0`` is
        ``sum_n E_n exp(+i omega t_n)``, which is evaluated as an inverse DFT times a
        linear phase for the axis offset, so any ``t[0]`` is handled exactly and no
        assumption is made about where the pulse sits in the record. Only the
        non-negative frequencies are kept, and the zero-frequency bin is dropped: a
        spectral phase is meaningless there.

        A real array is the electric field. A complex array is taken as the analytic
        signal in the optics convention, so that a positive frequency means a positive
        frequency, as everywhere else at the interface.

        Parameters
        ----------
        t : numpy.ndarray
            Time samples, s, shape ``(n_data,)``, uniformly spaced and increasing.
        field : numpy.ndarray
            Measured field on ``t``, arbitrary units, shape ``(n_data,)``.
        energy : float
            Pulse energy to scale to, J.
        phase : array_like, optional
            Further Taylor coefficients about ``wavelength``, ``s**n`` at order ``n``.
        t0 : float, optional
            Further delay, s.
        wavelength : float or None, optional
            Reference wavelength of the Taylor phase, m.

        Returns
        -------
        DataPulse
            The pulse, holding the spectrum of the record.

        Raises
        ------
        PulseError
            If the arrays disagree in length, the axis is not uniform and increasing, or
            the record is too short.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.pulses import DataPulse
        >>> t = np.linspace(-100e-15, 100e-15, 1024)
        >>> field = np.exp(-((t / 20e-15) ** 2)) * np.cos(2.35e15 * t)
        >>> pulse = DataPulse.from_time(t, field, energy=1e-6)
        >>> float(round(pulse.reference_wavelength * 1e9))
        802.0
        """
        times = np.asarray(t, dtype=np.float64).reshape(-1)
        values = np.asarray(field, dtype=np.complex128).reshape(-1)
        if times.size != values.size:
            msg = (
                f"t and field must have the same length, got {times.size} and "
                f"{values.size}"
            )
            raise PulseError(msg)
        if times.size < _MIN_DATA_POINTS:
            msg = f"a time record needs at least {_MIN_DATA_POINTS} samples"
            raise PulseError(msg)
        steps = np.diff(times)
        # The record is resampled by a DFT, which has no meaning on a non-uniform axis.
        # A relative spread of 1e-9 is far below any real digitiser's jitter and far
        # above the round-off of a linspace.
        if steps[0] <= 0.0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
            msg = "t must be uniformly spaced and increasing"
            raise PulseError(msg)
        n_data = times.size
        transform = n_data * np.fft.ifft(values)
        omega = 2.0 * np.pi * np.fft.fftfreq(n_data, float(steps[0]))
        keep = omega > 0.0
        return cls(
            omega=omega[keep],
            spectrum=transform[keep] * np.exp(1j * omega[keep] * times[0]),
            energy=energy,
            phase=phase,
            t0=t0,
            wavelength=wavelength,
        )


# --- continuous wave ------------------------------------------------------------------


@dataclass(frozen=True, eq=False, kw_only=True)
class CWPulse(Displayed):
    """A continuous-wave field, monochromatic or with a ``sech**2`` power spectrum.

    The spectrum is written straight onto the grid's own bins, so the field is periodic
    in the time window by construction and needs no apodisation; the average power is
    then realised in one pass over the flat part of the temporal window, rather than by
    alternating between a fixed spectral magnitude and a windowed field until the two
    agree. The single pass trades a slightly perturbed spectral shape for an exact power
    and a bounded cost.

    Attributes
    ----------
    wavelength : float
        Centre wavelength, m. Must fall inside the grid's active band.
    average_power : float
        Average power, W, over the samples where the temporal collar is exactly one.
    bandwidth : float
        Full width at half maximum of the power spectrum, in **wavelength**, m. Zero,
        the default, gives a monochromatic field on the single nearest bin. It is mapped
        to frequency at first order, ``Δω = 2πcΔλ/λ²`` at the centre wavelength: the
        ``sech**2`` spectrum has that frequency FWHM, the frequency equivalent of the
        requested width, and its wavelength FWHM is ``bandwidth`` only in the
        narrowband limit.
    seed : int or None
        Seed of the uniform random spectral phase used when ``bandwidth`` is positive,
        so that a realisation is reproducible. Unused for a monochromatic field.

    Raises
    ------
    PulseError
        If the wavelength, power or bandwidth is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.pulses import CWPulse
    >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (900e-9, 1.2e-6), 20e-12)
    >>> spectrum = CWPulse(wavelength=1030e-9, average_power=5.0).build(grid)
    >>> spectrum.shape[:2]
    (1, 1)
    """

    wavelength: float
    average_power: float
    bandwidth: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate the wavelength, the power and the bandwidth.

        Raises
        ------
        PulseError
            If the wavelength or power is not positive, or the bandwidth is negative or
            not finite.
        """
        _check_positive("wavelength", self.wavelength)
        _check_positive("average_power", self.average_power)
        if not math.isfinite(self.bandwidth) or self.bandwidth < 0.0:
            msg = (
                "bandwidth must be a finite non-negative wavelength width in metres, "
                f"got {self.bandwidth!r}"
            )
            raise PulseError(msg)

    def __repr__(self) -> str:
        """Return a one-line description.

        Returns
        -------
        str
            For example ``CWPulse(wavelength=1.064 um, average_power=10 W)``.
        """
        fields: list[tuple[str, object]] = [
            ("wavelength", engineering(self.wavelength, "m")),
            ("average_power", engineering(self.average_power, "W")),
        ]
        if self.bandwidth:
            fields.append(("bandwidth", engineering(self.bandwidth, "m")))
            fields.append(("seed", self.seed))
        return line("CWPulse", fields)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "CWPulse: a continuous-wave field", [
            ("wavelength", engineering(self.wavelength, "m")),
            ("average power", engineering(self.average_power, "W")),
            ("bandwidth", engineering(self.bandwidth, "m")),
            ("phase seed", self.seed),
        ]

    def build(self, grid: TimeGrid) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the continuous-wave spectrum on the given grid.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            Grid to build on.

        Returns
        -------
        numpy.ndarray
            Stored spectrum, shape ``(1, 1, n_omega)``, in the square root of watts.

        Raises
        ------
        PulseError
            If the centre wavelength falls outside the grid's active band.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop import fourier
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.pulses import CWPulse
        >>> grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (900e-9, 1.2e-6), 20e-12)
        >>> spectrum = CWPulse(wavelength=1030e-9, average_power=5.0).build(grid)
        >>> samples = fourier.inverse(spectrum[0, 0], kind=grid.kind)
        >>> flat = grid.t_window == 1.0
        >>> bool(np.isclose(np.mean(np.abs(samples[flat]) ** 2), 5.0, rtol=1e-12))
        True
        """
        omega_p = 2.0 * np.pi * C / self.wavelength
        active = grid.active_mask
        if not (grid.omega[active].min() <= omega_p <= grid.omega[active].max()):
            msg = (
                f"the centre wavelength {self.wavelength:.4g} m lies outside the "
                "grid's active band"
            )
            raise PulseError(msg)
        optics = np.zeros(grid.n_omega, dtype=np.complex128)
        if self.bandwidth > 0.0:
            # A wavelength FWHM maps to an angular-frequency FWHM by 2 pi c dlambda /
            # lambda**2, and a sech**2 power spectrum has
            # the same FWHM-to-natural-width ratio as a sech**2 pulse.
            width = (
                2.0 * np.pi * C * self.bandwidth / self.wavelength**2
            ) * SECH_FWHM_TO_WIDTH
            scaled = np.minimum(np.abs((grid.omega - omega_p) / width), _SECH_CLIP)
            rng = np.random.default_rng(self.seed)
            optics = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, grid.n_omega)) / np.cosh(
                scaled
            )
        else:
            distance = np.where(active, np.abs(grid.omega - omega_p), np.inf)
            optics[int(np.argmin(distance))] = 1.0
        optics[~active] = 0.0
        spectrum = fourier.spectrum_to_internal(optics, kind=grid.kind)
        return _canonical(_rescale(spectrum, grid, self.average_power, "average_power"))

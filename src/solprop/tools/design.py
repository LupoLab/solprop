# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later


"""Design solvers: the parameter of a guide or fill that meets a design target.

:func:`solve_parameter` brackets every root of a target's residual over a parameter;
:func:`radius_zdw`, :func:`pressure_zdw` and :func:`pressure_rdw` are its capillary
forms.
"""

import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, cast

import numpy as np
from jaxtyping import Float
from scipy.optimize import brentq

from solprop._validation import positive
from solprop.constants import ROOM_TEMPERATURE
from solprop.exceptions import (
    ToolsError,
)
from solprop.modes import base
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.modes.capillary import MarcatiliMode, MarcatiliModel
from solprop.modes.protocol import Mode
from solprop.physdata import (
    Gas,
    Mixture,
)
from solprop.physdata.glasses import GlassIndex
from solprop.physdata.selection import MaterialSelection
from solprop.pulses import GaussianPulse, SechPulse
from solprop.tools.phase_matching import (
    NonlinearPhase,
    _PhasePulse,
    lambda_rdw_all,
    phase_mismatch,
)
from solprop.tools.pulse import _SECH_WIDTH, PulseShape, _omega, pulse_peak_power
from solprop.tools.soliton import _area_per_order, nonlinear_coefficient, soliton_order
from solprop.tools.waveguide import (
    _ZDW_RANGE,
    _check_carrier,
    _default_cladding,
    _fill_n2,
    _probing,
    _resolve_fill,
    _zdw_near,
)

__all__ = [
    "RDW",
    "ZDW",
    "DesignBuild",
    "DesignRoots",
    "DesignTarget",
    "RootChoice",
    "SolitonOrder",
    "pressure_rdw",
    "pressure_zdw",
    "radius_zdw",
    "solve_parameter",
]

logger = logging.getLogger(__name__)


_PRESSURE_FLOOR = 1e-6
"""Lower end of the pressure brackets in bar; a true vacuum has no gas term."""


_SEARCH_SAMPLES = 129
"""Linear samples of a parameter search, and logarithmic ones beside them where the
range is positive: :func:`pressure_rdw`'s grid, which :func:`solve_parameter` shares."""


_ATTAINABLE_SAMPLES = 17
"""Linear samples, and logarithmic ones beside them, on which a search without a root
reports the attainable range of its target's quantity. Coarser than the search: the
report rebuilds each mode, and the zero-dispersion search alone costs some tens of
index evaluations per mode. Both ends of the range are among them, which is where a
quantity monotonic in the parameter takes its extremes."""


_ROOT_TOLERANCE = 2e-12
"""Brent's absolute tolerance on a bracketed root, in units of the bracket's larger
end: the default ``xtol`` of :func:`scipy.optimize.brentq` (2e-12) made relative, so a
radius in m and a pressure in bar converge to the same number of digits."""


_ROOT_RESIDUAL = 1e-3
"""Largest residual at a refined sign change, as a fraction of its larger value at the
bracket's ends, for the change to be a root. Brent's method converges onto a pole as
readily as onto a zero; at a zero the residual it ends on is of the order of the slope
times the tolerance, many orders below the ends, and at a pole it diverges past both.
A parameter has no integral like ``beta1`` to test the change against (see
:data:`~solprop.tools.waveguide._POLE_FACTOR`), so the residual's own size decides."""


def _pressure_root(
    residual: Callable[[float], float], pressure_max: float, what: str
) -> float:
    """Return the pressure at which a function of pressure vanishes.

    - ``residual``: a function of the pressure in bar.
    - ``pressure_max``: upper end of the bracket in bar.
    - Returns: the root in bar.
    """
    top = positive(pressure_max, "pressure_max", ToolsError)
    if top <= _PRESSURE_FLOOR:
        raise ToolsError(f"pressure_max must exceed {_PRESSURE_FLOOR} bar, got {top}")
    with _probing():
        low, high = residual(_PRESSURE_FLOOR), residual(top)
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ToolsError(
            f"the residual for {what} is not finite at the ends of the pressure "
            f"bracket ({low!r}, {high!r}): the index model does not reach that far"
        )
    if low * high > 0.0:
        raise ToolsError(
            f"no pressure between {_PRESSURE_FLOOR} and {top:.4g} bar gives {what}: "
            f"the residual is {low:.4g} at the low end and {high:.4g} at the high end"
        )
    with _probing():
        root = float(brentq(residual, _PRESSURE_FLOOR, top))
    # Outside the probing context, so the answer's own warnings are raised.
    residual(root)
    return root


def _design_capillary(
    radius: float,
    gas: Gas | str,
    *,
    temperature: float,
    m: int,
    model: MarcatiliModel,
    cladding: GlassIndex | complex | None,
    lattice: TubeLattice | None,
    materials: MaterialSelection | None,
) -> MarcatiliMode:
    """Build the single-fill capillary a pressure search re-evaluates.

    The mode `prop_capillary` builds with the same ``materials=``: the gas (or
    ``"air"``) as an optmat fill at one bar and ``cladding=None`` as optmat's
    recommended silica.

    - ``radius``: core radius in m, > 0.
    - ``temperature``: fill temperature in K.
    - ``cladding``: cladding index; ``None`` is fused silica.
    - ``lattice``: tube geometry for a VincettiMode; None builds a Marcatili capillary.
    """
    bore = positive(radius, "radius", ToolsError)
    wall = _default_cladding(cladding)
    # The fill is an optmat mixture, which carries its own pressure and temperature.
    mixture = _resolve_fill(gas, 1.0, temperature, materials)
    if lattice is None:
        return MarcatiliMode(
            bore, None, 0.0, m=m, model=model, cladding=wall, mixture=mixture
        )
    return VincettiMode(
        bore,
        None,
        0.0,
        m=m,
        model=model,
        cladding=wall,
        mixture=mixture,
        lattice=lattice,
    )


def _capillary_at(mode: MarcatiliMode, pressure: float) -> MarcatiliMode:
    """Return a single-fill capillary re-evaluated at another pressure.

    The mode is built once and frozen at each density through its parameterised form,
    so a root finder costs an equation-of-state call and an index evaluation per step
    rather than a mode construction. A selected fill takes its density from its
    species' own equation of state (optmat's); a selected air, two populations at
    one shared pressure, is frozen at that pressure.

    - ``pressure``: pressure in bar.
    """
    # pressure_rdw and pressure_zdw build this mode at their one scalar temperature.
    temperature = cast("float", mode.temperature)
    # `_design_capillary` always builds the mode on an optmat fill.
    fill = cast("Mixture", mode.mixture)
    if len(fill.components) > 1:
        return mode.at_parameters({"pressure": float(pressure)})
    state = next(iter(fill.components)).species.equation_of_state
    rho = float(state.density(float(pressure), temperature))
    return mode.at_parameters({"density": rho})


def _search_grid(
    low: float, high: float, samples: int = _SEARCH_SAMPLES
) -> Float[np.ndarray, " n_samples"]:
    """Return the values a sampled root search evaluates its residual at.

    - Returns: `samples` linear values, and as many logarithmic ones where ``low > 0``,
      merged and ascending; both ends are among them.
    """
    # The logarithmic grid resolves the low end of a range spanning decades (a
    # pressure from 1e-6 bar); the linear grid resolves structure at its high end.
    # Pairs of roots, or admissible intervals, narrower than a sample are not
    # guaranteed: narrowing the range resolves them.
    linear = np.linspace(low, high, samples)
    if low <= 0.0:
        return linear
    return np.unique(np.concatenate((np.geomspace(low, high, samples), linear)))


@dataclass(frozen=True)
class _Sampled:
    """The outcome of a sampled root search.

    Attributes
    ----------
    roots : tuple of float
        The roots, ascending.
    values : tuple of float
        The sampled values of the parameter, ascending.
    residuals : tuple of float or None
        The residual at each sample; None where the target is undefined.
    """

    roots: tuple[float, ...]
    values: tuple[float, ...]
    residuals: tuple[float | None, ...]


def _sampled_roots(
    residual: Callable[[float], float | None],
    limits: tuple[float, float],
    *,
    what: str,
    narrow: str,
) -> _Sampled:
    """Find every root of a residual over a range, without crossing a pole.

    The residual is sampled on :func:`_search_grid`; each sign change between two
    samples where it is defined is refined with Brent's method, and the refined
    point is a root only if the residual there is below :data:`_ROOT_RESIDUAL` of
    its larger value at the bracket's ends. A sample that is exactly zero is a root.

    - ``residual``: a function of the parameter returning a finite float, or None where
      the target is undefined (an inadmissible state, such as no soliton).
    """
    low, high = limits
    grid = _search_grid(low, high)
    # Brent's method re-evaluates the ends of its bracket and the root it returns,
    # which are samples already: one evaluation per distinct value.
    seen: dict[float, float | None] = {}

    def checked(value: float) -> float | None:
        """Evaluate the residual once per value, refusing a value that is not finite.

        - Returns: finite residual, or None where the target is undefined.
        """
        if value not in seen:
            result = residual(value)
            if result is not None and not math.isfinite(result):
                raise ToolsError(f"{what} is not finite at {value:g}")
            seen[value] = result
        return seen[value]

    def refine(value: float) -> float:
        """Evaluate the residual inside one bracket, where it must be defined."""
        result = checked(value)
        if result is None:
            # The only undefined state a target has is the largest constituent's
            # absence, below the first soliton threshold or in normal dispersion.
            raise ToolsError(
                f"the refinement of {what} crossed a no-soliton region at "
                f"{value:g}, between two samples outside it; narrow {narrow}"
            )
        return result

    with _probing():
        sampled = [checked(float(value)) for value in grid]
        roots = [
            float(value)
            for value, result in zip(grid, sampled, strict=True)
            if result == 0.0
        ]
        for i, (left, right) in enumerate(pairwise(sampled)):
            if left is None or right is None or left * right >= 0.0:
                continue
            start, stop = float(grid[i]), float(grid[i + 1])
            root = float(
                brentq(
                    refine,
                    start,
                    stop,
                    xtol=_ROOT_TOLERANCE * max(abs(start), abs(stop)),
                )
            )
            if abs(refine(root)) <= _ROOT_RESIDUAL * max(abs(left), abs(right)):
                roots.append(root)
            else:
                logger.debug(
                    "discarded the sign change of %s at %g as a pole", what, root
                )
    return _Sampled(
        tuple(sorted(roots)),
        tuple(float(value) for value in grid),
        tuple(sampled),
    )


def _rdw_pressure_root(
    residual: Callable[[float], float | None], limits: tuple[float, float]
) -> float:
    """Find a unique pressure root without crossing a sampled inadmissible region.

    - ``residual``: mismatch in 1/m, or None only where no focusing discrete soliton
      exists.
    - ``limits``: positive ordered pressure bracket in bar.
    - Returns: unique pressure root in bar.
    """
    low, high = limits
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
        raise ToolsError("pressure_limits must be finite, positive and ordered")
    roots = _sampled_roots(
        residual, limits, what="the RDW pressure residual", narrow="pressure_limits"
    ).roots
    if not roots:
        raise ToolsError(
            f"no pressure between {low:g} and {high:g} bar gives an admissible "
            "RDW root; narrow pressure_limits if needed"
        )
    root, *others = roots
    if others:
        found = ", ".join(f"{value:.4g}" for value in roots)
        raise ToolsError(
            f"ambiguous RDW pressure: {len(roots)} roots, at {found} bar; "
            "choose one with pressure_limits"
        )
    # Re-evaluate the accepted answer outside the probing warning filter.
    residual(root)
    return root


def pressure_rdw(
    radius: float,
    gas: Gas | str,
    target_wavelength: float,
    wavelength: float,
    *,
    pressure_max: float = 100.0,
    peak_power: float | None = None,
    fwhm: float | None = None,
    shape: PulseShape = "sech",
    nonlinear_phase: NonlinearPhase = "largest",
    lattice: TubeLattice | None = None,
    pressure_limits: tuple[float, float] | None = None,
    temperature: float = ROOM_TEMPERATURE,
    m: int = 1,
    model: MarcatiliModel = "full",
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
) -> float:
    """Return the pressure that phase matches a dispersive wave at a target wavelength.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    gas : Gas or str
        A single built-in gas.
    target_wavelength : float
        Wavelength the dispersive wave should have, m.
    wavelength : float
        Pump wavelength in m.
    pressure_max : float, optional
        Upper end of the pressure bracket in **bar**; the lower end is 1e-6 bar.
    peak_power : float or None, optional
        Peak power in W. When given, the soliton's nonlinear wavenumber is included
        and its nonlinear coefficient follows the pressure.
    fwhm : float or None, optional
        Positive input power FWHM in s, required with power for the default phase.
    shape : {'sech', 'gauss'}, optional
        Unchirped input shape.
    nonlinear_phase : {'largest', 'supplied_power', 'linear'}, optional
        Largest constituent by default; supplied power preserves gamma P0/2
        without fwhm. Linear requires omitting pulse numbers.
    lattice : TubeLattice or None, optional
        Tube geometry for VincettiMode; None builds a Marcatili capillary.
    pressure_limits : tuple of float or None, optional
        Positive ordered search range in bar, within (1e-6, pressure_max).
        None uses that whole range. Searches 129 logarithmic and 129 linear
        samples, refining admissible sign changes; sub-grid root pairs or valid
        intervals can be missed. Narrow the range to resolve them.
    temperature : float, optional
        Fill temperature in K.
    m : int, optional
        Radial index of the HE_1m mode.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Design the capillary `prop_capillary` builds under the same
        ``materials=``: the gas from optmat (index, density, and the Kerr
        coefficient at `wavelength` for the nonlinear phase) and ``cladding=None``
        as optmat's recommended fused silica. ``None`` is
        ``MaterialSelection()``.

    Returns
    -------
    float
        Pressure in bar.

    Raises
    ------
    ToolsError
        If no pressure in the bracket matches, or the selection names a carrier
        other than `wavelength`.

    Examples
    --------
    >>> from solprop.tools import pressure_rdw
    >>> f"{pressure_rdw(125e-6, 'He', 199.75e-9, 800e-9, cladding=1.45):.2f}"
    '1.00'
    """
    # Validate before scanning; the nonlinear coefficient follows the pressure.
    design = RDW(
        wavelength,
        target_wavelength,
        peak_power=peak_power,
        fwhm=fwhm,
        shape=shape,
        nonlinear_phase=nonlinear_phase,
    )
    top = positive(pressure_max, "pressure_max", ToolsError)
    if top <= _PRESSURE_FLOOR:
        raise ToolsError(f"pressure_max must exceed {_PRESSURE_FLOOR} bar, got {top}")
    limits = (_PRESSURE_FLOOR, top) if pressure_limits is None else pressure_limits
    if limits[0] < _PRESSURE_FLOOR or limits[1] > top:
        raise ToolsError("pressure_limits must lie within (1e-6, pressure_max) bar")
    materials = MaterialSelection() if materials is None else materials
    _check_carrier(materials, wavelength)
    base_mode = _design_capillary(
        radius,
        gas,
        temperature=temperature,
        m=m,
        model=model,
        cladding=cladding,
        lattice=lattice,
        materials=materials,
    )

    def _residual(pressure: float) -> float | None:
        """Return the mismatch at the target for one pressure.

        - ``pressure``: pressure in bar.
        - Returns: mismatch in 1/m, or None when the constituent does not exist.
        """
        mode = _capillary_at(base_mode, pressure)
        gamma = None
        if peak_power is not None:
            assert mode.mixture is not None
            index = _fill_n2(mode.mixture, wavelength)[0]
            gamma = nonlinear_coefficient(mode, index, wavelength)
        return _rdw_residual(design, mode, gamma, 0.0)

    return _rdw_pressure_root(_residual, limits)


def pressure_zdw(
    radius: float,
    gas: Gas | str,
    zdw_wavelength: float,
    *,
    pressure_max: float = 100.0,
    temperature: float = ROOM_TEMPERATURE,
    m: int = 1,
    model: MarcatiliModel = "full",
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
) -> float:
    """Return the pressure that puts the zero-dispersion point at a wavelength.

    Parameters
    ----------
    radius : float
        Core radius in m, > 0.
    gas : Gas or str
        A single built-in gas.
    zdw_wavelength : float
        Wavelength at which the dispersion should vanish, m.
    pressure_max : float, optional
        Upper end of the pressure bracket in **bar**; the lower end is 1e-6 bar.
    temperature : float, optional
        Fill temperature in K.
    m : int, optional
        Radial index of the HE_1m mode.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Design the capillary `prop_capillary` builds under the same
        ``materials=``: the gas's index and density from optmat and
        ``cladding=None`` as optmat's recommended fused silica. ``None`` is
        ``MaterialSelection()``.

    Returns
    -------
    float
        Pressure in bar.

    Raises
    ------
    ToolsError
        If no pressure in the bracket does.

    Examples
    --------
    >>> from solprop.tools import pressure_zdw
    >>> f"{pressure_zdw(125e-6, 'Ar', 500e-9, cladding=1.45):.3f}"
    '0.066'
    """
    materials = MaterialSelection() if materials is None else materials
    base_mode = _design_capillary(
        radius,
        gas,
        temperature=temperature,
        m=m,
        model=model,
        cladding=cladding,
        lattice=None,
        materials=materials,
    )
    omega = np.array([_omega(zdw_wavelength)])

    def _residual(pressure: float) -> float:
        """Return the dispersion at the wavelength for one pressure.

        - ``pressure``: pressure in bar.
        - Returns: ``beta_2`` in s^2/m, scaled to order one.
        """
        mode = _capillary_at(base_mode, pressure)
        # Scaled so that brentq's absolute tolerance on the residual is meaningless
        # and its tolerance on the pressure decides, as intended.
        return float(base.beta2(mode, omega)[0]) * 1e30

    return _pressure_root(
        _residual,
        pressure_max,
        f"a zero-dispersion wavelength of {zdw_wavelength * 1e9:.4g} nm",
    )


@dataclass(frozen=True)
class ZDW:
    r"""Target a zero-dispersion wavelength: :math:`\beta_2 = 0` at a wavelength.

    The residual :func:`solve_parameter` brackets is :math:`\beta_2` of the trial mode
    at `wavelength`, in s^2/m.

    Attributes
    ----------
    wavelength : float
        Wavelength at which the dispersion should vanish, m, > 0.

    Raises
    ------
    ToolsError
        If the wavelength is not finite and positive.

    Examples
    --------
    >>> from solprop.tools import ZDW
    >>> ZDW(500e-9)
    ZDW(wavelength=5e-07)
    """

    wavelength: float

    def __post_init__(self) -> None:
        """Refuse a wavelength that is not finite and positive.

        Raises
        ------
        ToolsError
            If the wavelength is not finite and positive.
        """
        positive(self.wavelength, "wavelength", ToolsError)


@dataclass(frozen=True)
class RDW:
    """Target a resonant dispersive wave phase matched at a wavelength.

    The residual :func:`solve_parameter` brackets is
    :func:`~solprop.tools.phase_matching.phase_mismatch` of the trial mode at `rdw`, in
    1/m, exactly as :func:`pressure_rdw` evaluates it: with `peak_power` the soliton's
    nonlinear wavenumber is included, with the trial mode's own nonlinear coefficient,
    and with the largest constituent (`fwhm` given) a trial value in normal dispersion
    or below the first soliton threshold has no residual and cannot bracket a root.

    Attributes
    ----------
    pump : float
        Wavelength of the soliton, m, > 0.
    rdw : float
        Wavelength the dispersive wave should have, m, > 0 and not the pump's.
    peak_power : float or None
        Input peak power in W for the nonlinear term; ``None`` is the linear
        condition. With it, the builder returns ``(mode, gamma)``.
    fwhm : float or None
        Input power FWHM in s; required with `peak_power` for the default phase.
    shape : {'sech', 'gauss'}
        Unchirped input shape for the largest constituent.
    nonlinear_phase : {'largest', 'supplied_power', 'linear'}
        The nonlinear phase model, as
        :func:`~solprop.tools.phase_matching.phase_mismatch` takes it.

    Raises
    ------
    ToolsError
        If a wavelength is not positive, the two are equal, or the pulse numbers
        are incomplete or contradictory for the phase model.

    Examples
    --------
    >>> from solprop.tools import RDW
    >>> RDW(800e-9, 200e-9, peak_power=2e9, fwhm=10e-15).nonlinear_phase
    'largest'
    """

    pump: float
    rdw: float
    peak_power: float | None = None
    fwhm: float | None = None
    shape: PulseShape = "sech"
    nonlinear_phase: NonlinearPhase = "largest"

    def __post_init__(self) -> None:
        """Refuse wavelengths or pulse numbers the phase matching does not define.

        Raises
        ------
        ToolsError
            If a wavelength is not positive, the two are equal, or the pulse
            numbers are incomplete or contradictory.
        """
        # A placeholder gamma: the real one comes with each trial mode.
        _PhasePulse(
            self.peak_power,
            None if self.peak_power is None else 1.0,
            self.fwhm,
            self.shape,
            self.nonlinear_phase,
        )
        positive(self.pump, "pump", ToolsError)
        positive(self.rdw, "rdw", ToolsError)
        if self.rdw == self.pump:
            # The linear mismatch and its slope vanish at the pump for every mode.
            raise ToolsError(
                f"the dispersive wave cannot be at the pump wavelength, {self.pump:g} m"
            )


@dataclass(frozen=True)
class SolitonOrder:
    """Target a conventional soliton order for a fixed pulse.

    The residual :func:`solve_parameter` brackets is
    :func:`~solprop.tools.soliton.soliton_order` of the pulse in the trial mode, with
    the trial mode's own nonlinear coefficient, minus `order`: a guide or fill parameter
    is solved for, the pulse is held. The order is the conventional measure on either
    side of a zero-dispersion point, so a search across one may find a root on each
    side. The opposite question, the pulse that has an order on a given mode, is closed
    form: :func:`~solprop.tools.soliton.soliton_peak_power` and
    :func:`~solprop.tools.pulse.pulse_energy`.

    Attributes
    ----------
    order : float
        Conventional soliton order, > 0.
    pulse : GaussianPulse or SechPulse
        The pulse: its wavelength is the pump, its FWHM and its energy or peak power
        are held, and its class sets the shape. An ordinary Gaussian (``m = 1``)
        with no spectral phase beyond a carrier-envelope phase and a delay.

    Raises
    ------
    ToolsError
        If the order is not positive, the Gaussian is super-Gaussian, or the pulse
        is chirped.

    Examples
    --------
    >>> from solprop.pulses import SechPulse
    >>> from solprop.tools import SolitonOrder
    >>> target = SolitonOrder(3.0, SechPulse(wavelength=800e-9, fwhm=10e-15,
    ...                                      energy=100e-6))
    >>> f"{target.peak_power / 1e9:.3f} GW"
    '8.814 GW'
    """

    order: float
    pulse: GaussianPulse | SechPulse

    def __post_init__(self) -> None:
        """Refuse an order or a pulse the design numbers are not defined for.

        Raises
        ------
        ToolsError
            If the order is not positive, the Gaussian is super-Gaussian, or the
            pulse is chirped.
        """
        positive(self.order, "order", ToolsError)
        if isinstance(self.pulse, GaussianPulse) and self.pulse.m != 1:
            raise ToolsError(
                "the soliton order is defined for sech and Gaussian pulses, and this "
                f"is a super-Gaussian of order m={self.pulse.m}"
            )
        if np.any(np.asarray(self.pulse.phase, dtype=float)[2:] != 0.0):
            # The design numbers take the width and peak power of an unchirped pulse.
            raise ToolsError(
                "the soliton order is defined for an unchirped pulse; this one has "
                "spectral phase beyond the carrier-envelope phase and the delay"
            )

    @property
    def shape(self) -> PulseShape:  # numpydoc ignore=RT01
        """Pulse shape the design numbers use.

        ``"gauss"`` for a `GaussianPulse`, ``"sech"`` for a `SechPulse`.
        """
        return "gauss" if isinstance(self.pulse, GaussianPulse) else "sech"

    @property
    def fwhm(self) -> float:  # numpydoc ignore=RT01
        """The pulse's full width at half maximum of the power, s.

        A `SechPulse` given its natural width converts it.
        """
        pulse = self.pulse
        if isinstance(pulse, GaussianPulse):
            return pulse.fwhm
        # SechPulse holds exactly one of the two widths (its own validation).
        return (
            pulse.fwhm
            if pulse.fwhm is not None
            else cast("float", pulse.width) * _SECH_WIDTH
        )

    @property
    def peak_power(self) -> float:  # numpydoc ignore=RT01
        """The pulse's peak power, W.

        The pulse's own, or that of its energy
        (:func:`~solprop.tools.pulse.pulse_peak_power`).
        """
        if self.pulse.peak_power is not None:
            return self.pulse.peak_power
        # ShapedPulse holds exactly one of energy and peak power.
        return pulse_peak_power(cast("float", self.pulse.energy), self.fwhm, self.shape)


DesignTarget = ZDW | RDW | SolitonOrder
"""A target of :func:`solve_parameter`."""


RootChoice = Literal["lowest", "highest"] | int
"""Which of several roots: the lowest, the highest, or an index into them."""


DesignBuild = Callable[[float], Mode | tuple[Mode, float]]
"""A builder: a trial value of the parameter to a mode, or to a mode and the fill's
nonlinear coefficient in rad/(W m)
(:func:`~solprop.tools.soliton.nonlinear_coefficient`)."""


@dataclass(frozen=True)
class DesignRoots:
    """Every root of a design target over a parameter's search range.

    Iterating gives the roots, and its length is their number.

    Attributes
    ----------
    roots : tuple of float
        The values of the parameter that meet the target, ascending, in the
        parameter's unit; empty when none does.
    search : tuple of float
        The search range, ``(low, high)``.
    target : ZDW, RDW or SolitonOrder
        The target.
    builds : int
        How many times the builder was called, the search's cost.
    quantity : str
        What `attainable` measures, with its unit: the zero-dispersion wavelength
        nearest the target's (m), the phase-matched wavelength nearest the target's
        (m), or the soliton order.
    attainable : tuple of float or None
        When there is no root, the smallest and largest `quantity` on a coarser grid
        of the search range (17 linear and, where ``low > 0``, 17 logarithmic
        values, the ends included), where it is defined: a mode with no zero
        dispersion in 100 nm to 3 um, or with no phase-matched wavelength in the
        RDW window or no soliton, has none. ``None`` when there are roots, or no
        value defines it.

    Examples
    --------
    >>> from solprop.tools import DesignRoots, ZDW
    >>> roots = DesignRoots((1.0, 2.0), (0.5, 3.0), ZDW(5e-7), 260,
    ...                     "zero-dispersion wavelength in m")
    >>> len(roots), roots.pick("highest"), roots.pick(0)
    (2, 2.0, 1.0)
    """

    roots: tuple[float, ...]
    search: tuple[float, float]
    target: DesignTarget
    builds: int
    quantity: str
    attainable: tuple[float, float] | None = None

    def __len__(self) -> int:
        """Return the number of roots.

        Returns
        -------
        int
            The number of roots.
        """
        return len(self.roots)

    def __iter__(self) -> Iterator[float]:
        """Iterate over the roots, ascending.

        Returns
        -------
        Iterator
            The roots.
        """
        return iter(self.roots)

    def pick(self, root: RootChoice = "lowest") -> float:
        """Return one root.

        Parameters
        ----------
        root : {'lowest', 'highest'} or int, optional
            The lowest or highest root, or an index into the ascending roots
            (negative from the end, as a sequence's).

        Returns
        -------
        float
            The root, in the parameter's unit.

        Raises
        ------
        ToolsError
            If there is no root, naming the attainable range, or the index is out of
            range, naming the roots.

        Examples
        --------
        >>> from solprop.tools import DesignRoots, ZDW
        >>> DesignRoots((1.0, 2.0), (0.5, 3.0), ZDW(5e-7), 260, "").pick(-1)
        2.0
        """
        if not self.roots:
            raise ToolsError(self._unreachable())
        if root == "lowest":
            return self.roots[0]
        if root == "highest":
            return self.roots[-1]
        if isinstance(root, bool) or not isinstance(root, int):
            raise ToolsError(
                f"root must be 'lowest', 'highest' or an index, got {root!r}"
            )
        count = len(self.roots)
        if not -count <= root < count:
            found = ", ".join(f"{value:.6g}" for value in self.roots)
            raise ToolsError(
                f"root index {root} is out of range: {count} root"
                f"{'' if count == 1 else 's'}, at {found}"
            )
        return self.roots[root]

    def _unreachable(self) -> str:
        """Return why no root was found."""
        low, high = self.search
        if self.attainable is None:
            seen = f"the {self.quantity} is undefined across it"
        else:
            seen = (
                f"across it the {self.quantity} spans "
                f"{self.attainable[0]:.6g} to {self.attainable[1]:.6g}"
            )
        return (
            f"no value between {low:.6g} and {high:.6g} gives "
            f"{_describe(self.target)}: {seen}"
        )


def _describe(target: DesignTarget) -> str:
    """Return a target in words, for messages.

    - Returns: for example ``"a zero-dispersion wavelength of 500 nm"``.
    """
    if isinstance(target, ZDW):
        return f"a zero-dispersion wavelength of {target.wavelength * 1e9:.6g} nm"
    if isinstance(target, RDW):
        return (
            f"a dispersive wave at {target.rdw * 1e9:.6g} nm from the pump at "
            f"{target.pump * 1e9:.6g} nm"
        )
    return f"a soliton order of {target.order:.6g}"


def _quantity_name(target: DesignTarget) -> str:
    """Return what a target's attainable range measures, with its unit."""
    if isinstance(target, ZDW):
        return "zero-dispersion wavelength nearest the target (m)"
    if isinstance(target, RDW):
        return "phase-matched wavelength nearest the target (m)"
    return "soliton order"


def _needs_gamma(target: DesignTarget) -> bool:
    """Return whether a target needs the fill's nonlinear coefficient."""
    if isinstance(target, RDW):
        return target.peak_power is not None
    return isinstance(target, SolitonOrder)


def _pump_dispersion(mode: Mode, wavelength: float, z: float) -> float:
    """Return the group-velocity dispersion at a pump, refusing one not finite.

    - ``wavelength``: pump wavelength in m.
    - ``z``: position along the waveguide, m.
    - Returns: ``beta_2`` in s^2/m.
    """
    dispersion = float(base.beta2(mode, np.array([_omega(wavelength)]), z)[0])
    if not math.isfinite(dispersion):
        raise ToolsError(
            f"the pump dispersion is not finite at {wavelength * 1e9:.6g} nm in a "
            "trial mode"
        )
    return dispersion


def _rdw_residual(
    target: RDW, mode: Mode, gamma: float | None, z: float
) -> float | None:
    """Return the phase mismatch at a dispersive-wave target, as the RDW search uses it.

    - ``gamma``: the mode's nonlinear coefficient in rad/(W m); given with a peak power.
    - ``z``: position along the waveguide, m.
    - Returns: mismatch in 1/m, or None when the largest constituent does not exist: in
      normal dispersion, or at a pulse area at or below the first threshold.
    """
    if target.peak_power is not None and target.fwhm is not None:
        dispersion = _pump_dispersion(mode, target.pump, z)
        if dispersion >= 0:
            return None
        order = soliton_order(
            target.peak_power,
            target.fwhm,
            cast("float", gamma),
            dispersion,
            target.shape,
        )
        if order * _area_per_order(target.shape) <= 0.5:
            return None
    return float(
        phase_mismatch(
            mode,
            np.array([_omega(target.rdw)]),
            target.pump,
            z=z,
            peak_power=target.peak_power,
            gamma=gamma,
            fwhm=target.fwhm,
            shape=target.shape,
            nonlinear_phase=target.nonlinear_phase,
        )[0]
    )


def _target_residual(
    target: DesignTarget, mode: Mode, gamma: float | None, z: float
) -> float | None:
    """Return a target's residual on one trial mode.

    - ``gamma``: its nonlinear coefficient in rad/(W m), where the target needs one.
    - ``z``: position along the waveguide, m.
    - Returns: ``beta_2`` in s^2/m, the mismatch in 1/m (None where no soliton exists),
      or the soliton order less the target's.
    """
    if isinstance(target, ZDW):
        return float(base.beta2(mode, np.array([_omega(target.wavelength)]), z)[0])
    if isinstance(target, RDW):
        return _rdw_residual(target, mode, gamma, z)
    dispersion = _pump_dispersion(mode, target.pulse.wavelength, z)
    order = soliton_order(
        target.peak_power, target.fwhm, cast("float", gamma), dispersion, target.shape
    )
    return order - target.order


def _rdw_window(target: RDW) -> tuple[float, float]:
    """Return the window an RDW target's attainable wavelengths are searched in.

    - Returns: :func:`~solprop.tools.phase_matching.lambda_rdw`'s default window, 100 nm
    to 0.9 of the pump, when the
      target is on the blue side of the pump, widened to include it; and the mirror of
      it, 1/0.9 of the pump to 3 um, on the red side.
    """
    if target.rdw < target.pump:
        return (min(_ZDW_RANGE[0], 0.9 * target.rdw), 0.9 * target.pump)
    return (target.pump / 0.9, max(_ZDW_RANGE[1], target.rdw / 0.9))


def _target_quantity(
    target: DesignTarget, mode: Mode, gamma: float | None, z: float
) -> float:
    """Return what a target measures on one trial mode, for the attainable range.

    - ``gamma``: its nonlinear coefficient in rad/(W m), where the target needs one.
    - ``z``: position along the waveguide, m.
    - Returns: the zero-dispersion or phase-matched wavelength nearest the target's in
      m, or the soliton order; NaN where the mode has none.
    """
    if isinstance(target, ZDW):
        return _zdw_near(mode, _omega(target.wavelength), z)
    if isinstance(target, SolitonOrder):
        residual = _target_residual(target, mode, gamma, z)
        return target.order + cast("float", residual)
    if _rdw_residual(target, mode, gamma, z) is None:
        return math.nan
    try:
        roots = lambda_rdw_all(
            mode,
            target.pump,
            z=z,
            peak_power=target.peak_power,
            gamma=gamma,
            fwhm=target.fwhm,
            shape=target.shape,
            nonlinear_phase=target.nonlinear_phase,
            wavelength_limits=_rdw_window(target),
        )
    except ToolsError:
        # The window reaches beyond the mode's index model; this is a diagnostic
        # for a target already found unreachable, so the sample has no wavelength.
        return math.nan
    if not roots:
        return math.nan
    return min(roots, key=lambda root: abs(root - target.rdw))


def _built(
    build: DesignBuild, value: float, target: DesignTarget
) -> tuple[Mode, float | None]:
    """Call a builder and check what it returned.

    - Returns: the mode, and the nonlinear coefficient in rad/(W m), or None where the
      target does not use one.
    """
    result = build(value)
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ToolsError(
                f"build({value:g}) returned a tuple of {len(result)}; give a mode "
                "or (mode, gamma)"
            )
        mode, gamma = result
        gamma = positive(gamma, "gamma", ToolsError)
    else:
        mode, gamma = result, None
    if not isinstance(mode, Mode):
        raise ToolsError(
            f"build({value:g}) returned {type(mode).__name__}, not a mode or "
            "(mode, gamma)"
        )
    if not _needs_gamma(target):
        # One builder serves every target: a linear target ignores the coefficient.
        return mode, None
    if gamma is None:
        raise ToolsError(
            f"{_describe(target)} needs the fill's nonlinear coefficient: build "
            "returns (mode, gamma), with gamma from nonlinear_coefficient"
        )
    return mode, gamma


def solve_parameter(
    build: DesignBuild,
    search: tuple[float, float],
    target: DesignTarget,
    *,
    z: float = 0.0,
) -> DesignRoots:
    """Return every value of a guide or fill parameter that meets a design target.

    The general form of :func:`pressure_zdw`, :func:`pressure_rdw` and
    :func:`radius_zdw`, for any mode and any parameter: a radius, a pressure, a core
    size or a wall thickness. The target's residual is sampled over `search` on 129
    linear values and, where ``low > 0``, 129 logarithmic ones (as
    :func:`pressure_rdw` samples pressure); each sign change between two samples is
    refined with Brent's method to a relative tolerance of 2e-12; and a refined
    change is kept as a root only where the residual has fallen to a thousandth of
    its value at the bracket's ends, which discards a change through a pole (a wall
    resonance, where it diverges instead). Roots closer together than one sample
    are not separated: narrow `search` to resolve them.

    Parameters
    ----------
    build : callable
        ``build(value)`` returns the mode at a trial value of the parameter, or ``(mode,
        gamma)`` with the fill's nonlinear coefficient in rad/(W m)
        (:func:`~solprop.tools.soliton.nonlinear_coefficient`), which an RDW target with
        a peak power and a soliton-order target need: a mode need not know its fill's
        Kerr response.
    search : tuple of float
        Finite ordered range of the parameter, ``(low, high)``, in its unit.
    target : ZDW, RDW or SolitonOrder
        What the parameter should achieve.
    z : float, optional
        Position along the waveguide at which each mode is evaluated, m: the launch
        position.

    Returns
    -------
    DesignRoots
        Every root, ascending, with the attainable range of the target's quantity
        when there is none; ``.pick(root)`` chooses one.

    Raises
    ------
    ToolsError
        If the range or position is not finite or not ordered, the builder returns
        something other than a mode (and a coefficient where one is needed), a
        residual is not finite, or a refinement crosses a no-soliton region.

    Examples
    --------
    The radius of a helium capillary at 1 bar with its zero dispersion at 400 nm,
    re-evaluating one mode at each trial radius:

    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.tools import ZDW, solve_parameter
    >>> mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    >>> roots = solve_parameter(
    ...     lambda radius: mode.at_parameters({"core_radius": radius}),
    ...     (20e-6, 500e-6),
    ...     ZDW(400e-9),
    ... )
    >>> [f"{radius * 1e6:.2f} um" for radius in roots]
    ['89.49 um']
    """
    low, high = (float(value) for value in search)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ToolsError(f"search must be finite and ordered, got {search!r}")
    if not math.isfinite(z):
        raise ToolsError(f"z must be finite, got {z!r}")
    builds = 0

    def _residual(value: float) -> float | None:
        """Build the mode at one value and return the target's residual.

        - Returns: the residual, or None where the target is undefined.
        """
        nonlocal builds
        builds += 1
        mode, gamma = _built(build, value, target)
        return _target_residual(target, mode, gamma, z)

    sampled = _sampled_roots(
        _residual, (low, high), what="the target's residual", narrow="search"
    )
    # Outside the probing context, so a root beyond an index model's range warns.
    for root in sampled.roots:
        _residual(root)
    attainable = None
    if not sampled.roots:
        # The modes are rebuilt rather than kept from the search, on a coarser grid:
        # a tabulated mode can be large, and only a search without a root pays.
        with _probing():
            values = []
            for value in _search_grid(low, high, _ATTAINABLE_SAMPLES):
                builds += 1
                mode, gamma = _built(build, float(value), target)
                values.append(_target_quantity(target, mode, gamma, z))
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            attainable = (min(finite), max(finite))
    return DesignRoots(
        roots=sampled.roots,
        search=(low, high),
        target=target,
        builds=builds,
        quantity=_quantity_name(target),
        attainable=attainable,
    )


def radius_zdw(
    gas: Gas | str,
    pressure: float,
    zdw_wavelength: float,
    *,
    radius_limits: tuple[float, float] = (5e-6, 1e-3),
    temperature: float = ROOM_TEMPERATURE,
    m: int = 1,
    model: MarcatiliModel = "full",
    cladding: GlassIndex | complex | None = None,
    materials: MaterialSelection | None = None,
    root: RootChoice = "lowest",
) -> float:
    """Return the core radius that puts a capillary's zero dispersion at a wavelength.

    The radius counterpart of :func:`pressure_zdw`, through
    :func:`solve_parameter`: the capillary is built once at `pressure` and
    re-evaluated at each trial radius.

    Parameters
    ----------
    gas : Gas or str
        A single built-in gas.
    pressure : float
        Fill pressure in **bar**, > 0.
    zdw_wavelength : float
        Wavelength at which the dispersion should vanish, m.
    radius_limits : tuple of float, optional
        Positive ordered search range of the core radius, m.
    temperature : float, optional
        Fill temperature in K.
    m : int, optional
        Radial index of the HE_1m mode.
    model : {'full', 'reduced'}, optional
        Marcatili model.
    cladding : GlassIndex, complex or None, optional
        Cladding index; ``None`` is fused silica.
    materials : MaterialSelection or None, optional
        Design the capillary `prop_capillary` builds under the same
        ``materials=``, as :func:`pressure_zdw` does. ``None`` is
        ``MaterialSelection()``.
    root : {'lowest', 'highest'} or int, optional
        Which root, should there be several (``DesignRoots.pick``).

    Returns
    -------
    float
        Core radius in m.

    Raises
    ------
    ToolsError
        If the radius limits are not positive and ordered, the pressure is not
        positive, or no radius in the range gives the zero (naming the range of
        zero-dispersion wavelengths it does give).

    Examples
    --------
    >>> from solprop.tools import radius_zdw
    >>> f"{radius_zdw('He', 1.0, 400e-9, cladding=1.45) * 1e6:.2f} um"
    '89.49 um'
    """
    low, high = radius_limits
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ToolsError(
            f"radius_limits must be finite, positive and ordered, got {radius_limits!r}"
        )
    materials = MaterialSelection() if materials is None else materials
    capillary = _capillary_at(
        _design_capillary(
            high,
            gas,
            temperature=temperature,
            m=m,
            model=model,
            cladding=cladding,
            lattice=None,
            materials=materials,
        ),
        positive(pressure, "pressure", ToolsError),
    )
    roots = solve_parameter(
        lambda radius: capillary.at_parameters({"core_radius": radius}),
        (low, high),
        ZDW(zdw_wavelength),
    )
    return roots.pick(root)

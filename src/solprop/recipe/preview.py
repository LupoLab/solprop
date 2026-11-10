# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Previews: what a recipe will build, shown before the run (SEP 41, S18).

:func:`modes` gives the mode set a guided route builds, with its effective index,
dispersion to third order, loss, effective area and nonlinear coefficient over the
grid's band, and each mode's phase matching to a soliton at the pump in the launch
mode; :func:`preview_input` gives the input pulse on the
run's grid as the launch sets it, without noise, through the same processing functions
that read a run's file, and on the free-space routes the beam's fluence across the
launch plane too. Both resolve the recipe's pins first.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop._host import ObserverSpend
from solprop.exceptions import ModeError, ToolsError
from solprop.recipe.errors import RecipeError

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation
    from solprop.grid import FreeGrid, RadialGrid
    from solprop.modes.protocol import Mode
    from solprop.modes.slab import PlanarGrid
    from solprop.recipe.core import Recipe
    from solprop.spatial import SpatialMedium

__all__ = [
    "GUIDED",
    "InputPreview",
    "ModePreview",
    "ModeSetPreview",
    "modes",
    "preview_input",
]

GUIDED = ("capillary", "waveguide", "planar", "gnlse")
"""The routes with a mode set to preview.

The planar route's mode set is its grid's slab channels in x, the modes the beam is
projected onto, at ``k_y = 0``; the GNLSE's is its one specified mode, from its Taylor
coefficients or its fibre, without a fill (so without ``gamma`` over the band).
"""


@dataclass(frozen=True)
class ModePreview:
    """One mode of a guided route, over the grid's band.

    Attributes
    ----------
    label : str
        Its display label, as ``launch`` and ``channel`` name it.
    neff_minus_one : numpy.ndarray
        The real part of the effective index less one, formed without cancellation.
    beta : numpy.ndarray
        Propagation constant in 1/m.
    beta1 : numpy.ndarray
        Group delay per unit length in s/m, analytic.
    beta2 : numpy.ndarray
        Group-velocity dispersion in s^2/m.
    beta3 : numpy.ndarray
        Third-order dispersion in s^3/m (:func:`solprop.modes.base.beta3`).
    alpha : numpy.ndarray
        Power loss coefficient in 1/m.
    aeff : float
        Effective area in m^2; for a slab mode, an effective width in m.
    zdw : float or None
        Zero-dispersion wavelength in m within the band, or ``None`` where the
        dispersion keeps one sign across it (or changes it more than once).
    gamma : numpy.ndarray or None
        The nonlinear coefficient ``n2 omega / (c A_eff)`` in rad/(W m), with the
        fill's ``n2`` at the grid's reference wavelength, the carrier its Kerr data
        are taken at (:func:`solprop.tools.nonlinear_coefficient` at each
        frequency); ``None`` where the mode has no fill or the fill no Kerr data.
    phase_mismatch : numpy.ndarray or None
        The phase mismatch in rad/m between a dispersive wave in this mode and a
        soliton in the launch mode (the set's first) at the pump
        (:func:`solprop.tools.phase_mismatch`); ``None`` without a pump. See
        :func:`modes` for the pump and the nonlinear term.
    rdw : tuple of float
        The phase-matched wavelengths in m, shortest first, between the band's
        short edge and 0.9 of the pump (:func:`solprop.tools.lambda_rdw_all`);
        empty without a pump or a root.
    """

    label: str
    neff_minus_one: Float[np.ndarray, " n_omega"]
    beta: Float[np.ndarray, " n_omega"]
    beta1: Float[np.ndarray, " n_omega"]
    beta2: Float[np.ndarray, " n_omega"]
    beta3: Float[np.ndarray, " n_omega"]
    alpha: Float[np.ndarray, " n_omega"]
    aeff: float
    zdw: float | None
    gamma: Float[np.ndarray, " n_omega"] | None
    phase_mismatch: Float[np.ndarray, " n_omega"] | None = None
    rdw: tuple[float, ...] = field(default=())


@dataclass(frozen=True)
class ModeSetPreview:
    """The mode set a guided route builds, at one position.

    Attributes
    ----------
    z : float
        The position, m.
    omega : numpy.ndarray
        The in-band angular frequencies, rad/s, ascending.
    modes : tuple of ModePreview
        The modes, in the order the run holds them.
    pump : float or None
        The pump wavelength the phase matching is for, m: the recipe's
        ``wavelength``; ``None`` when it gives none (a recipe of ``pulses``).
    nonlinear : bool
        Whether the phase matching has the soliton's nonlinear term: the recipe's
        pulse is an unchirped sech or Gaussian, the launch mode's fill has Kerr
        data and the pump is in anomalous dispersion; otherwise it is the linear
        condition.
    """

    z: float
    omega: Float[np.ndarray, " n_omega"]
    modes: tuple[ModePreview, ...]
    pump: float | None = None
    nonlinear: bool = False


@dataclass(frozen=True)
class InputPreview:
    """The input pulse on the run's grid, as the launch sets it, without noise.

    On a free-space route (``radial``, ``free``, ``planar``) the power and the energy
    are the beam's, the axial flux through the launch plane in the medium
    (:func:`~solprop.spatial.spatial_power`, :func:`~solprop.spatial.spatial_energy`),
    and the fluence is its time integral at each transverse node
    (:func:`~solprop.spatial.spatial_fluence`, or
    :func:`~solprop.spatial.planar_fluence` on the planar route). On a line grid the
    power and the energy are per metre of y, W/m and J/m.

    Attributes
    ----------
    time : numpy.ndarray
        The time axis, s.
    power : numpy.ndarray
        The instantaneous power, W, summed over modes or integrated over the beam.
    omega : numpy.ndarray
        The angular frequency axis, rad/s.
    spectrum : numpy.ndarray
        The complex spectrum, per mode or per transverse column, as
        :func:`~solprop.processing.fields.get_spectrum` gives it.
    energy : float
        The pulse energy, J.
    peak_power : float
        The peak power, W.
    fluence : numpy.ndarray or None
        The beam's fluence at the launch plane, J/m^2, on the free-space routes: shape
        ``(n_r,)`` on a radial grid, ``(n_x,)`` on a line, ``(n_y, n_x)`` on a
        Cartesian grid and ``(n_y, n_x)`` on the planar route's (x, y) nodes;
        ``None`` on a guided route.
    axes : tuple of numpy.ndarray
        The fluence's axes in m, one per dimension in its order: ``(r,)``,
        ``(x,)`` or ``(y, x)``; empty on a guided route.
    """

    time: Float[np.ndarray, " n_t"]
    power: Float[np.ndarray, " n_t"]
    omega: Float[np.ndarray, " n_omega"]
    spectrum: Complex[np.ndarray, "*modes n_omega"]
    energy: float
    peak_power: float
    fluence: Float[np.ndarray, "*nodes"] | None = None
    axes: tuple[Float[np.ndarray, " _axis"], ...] = ()


def modes(recipe: "Recipe", z: float | None = None) -> ModeSetPreview:  # noqa: UP037 - run time
    """Return the mode set a guided recipe builds, without building its run.

    Its cost is the mode solve: on a guide route, the dense tables.

    Parameters
    ----------
    recipe : Recipe
        A recipe on a guided route.
    z : float or None, optional
        The position, m; ``None`` for the launch position.

    Returns
    -------
    ModeSetPreview
        The modes over the grid's band.

    Raises
    ------
    RecipeError
        If the route is not one of :data:`GUIDED`.

    Notes
    -----
    The phase matching is that of a soliton at the recipe's pump ``wavelength``
    travelling in the launch mode, the set's first, and a dispersive wave in each
    mode (the launch mode's own for the first; intermodal for the rest). With an
    unchirped sech or Gaussian pulse given by its FWHM and its energy or peak power,
    and a fill with Kerr data, the soliton's nonlinear wavenumber is that of its
    largest constituent (:func:`solprop.tools.phase_mismatch`'s default); otherwise,
    and for a pump in normal dispersion, which launches no soliton, the condition is
    the linear one.
    """
    from solprop.modes.base import (  # noqa: PLC0415
        aeff,
        alpha,
        beta,
        beta1,
        beta2,
        beta3,
        zdw,
    )

    if recipe.route not in GUIDED:
        msg = f"the {recipe.route} route builds no mode set"
        raise RecipeError(msg)
    simulation = recipe.build()
    grid = simulation.grid
    channels = (
        cast("PlanarGrid", simulation.spatial_grid).modes
        if recipe.route == "planar"
        else simulation.modes
    )
    omega = np.sort(np.asarray(grid.omega)[grid.inband_mask])
    at = float(simulation.initial_z or 0.0) if z is None else float(z)
    limits = (
        2 * np.pi * _C / float(omega[-1]),
        2 * np.pi * _C / float(omega[0]),
    )
    keywords = recipe.resolve().keywords()
    pump, pulse = _pump(keywords, channels[0] if channels else None, at)
    previews = []
    for mode in channels:
        try:
            zero: float | None = zdw(mode, at, wavelength_limits=limits)
        except ModeError:
            zero = None
        area = aeff(mode, at)
        mismatch, roots = _phase_matching(
            mode, channels[0], omega, z=at, pump=pump, pulse=pulse, limits=limits
        )
        previews.append(
            ModePreview(
                label=mode.channel_label.display_label,
                neff_minus_one=np.real(mode.neff_minus_one(omega, at)),
                beta=beta(mode, omega, at),
                beta1=beta1(mode, omega, at),
                beta2=beta2(mode, omega, at),
                beta3=beta3(mode, omega, at),
                alpha=alpha(mode, omega, at),
                aeff=area,
                zdw=zero,
                gamma=_gamma(mode, omega, area, grid.reference_wavelength, at),
                phase_mismatch=mismatch,
                rdw=roots,
            )
        )
    return ModeSetPreview(at, omega, tuple(previews), pump, bool(pulse))


_C = 299_792_458.0


def _pump(  # noqa: PLR0911 - one return per case of the linear condition
    keywords: Mapping[str, object],
    soliton: "Mode | None",  # noqa: UP037 - TYPE_CHECKING only
    z: float = 0.0,
) -> tuple[float | None, dict[str, object]]:
    """Return the pump wavelength and the soliton's nonlinear numbers, where given.

    Parameters
    ----------
    keywords : collections.abc.Mapping
        The resolved recipe's route keywords.
    soliton : solprop.modes.base.Mode or None
        The launch mode.
    z : float, optional
        The position, m: the fill's coefficient, the mode's area and its dispersion
        are taken there, as the mismatch is.

    Returns
    -------
    tuple
        The pump wavelength in m (``None`` without a numeric ``wavelength``), and
        the keywords :func:`solprop.tools.phase_mismatch` takes for the nonlinear
        term (``peak_power``, ``gamma``, ``fwhm``, ``shape``), empty for the linear
        condition. The nonlinear coefficient is the recipe's ``gamma`` where it
        gives one (the GNLSE), else the launch mode's fill's at `z`. The condition
        is linear for a chirped pulse (a ``phases`` term from second order on), a
        pump in normal dispersion, and a pulse below the soliton threshold, which
        has no largest constituent.
    """
    from solprop.modes.base import beta2  # noqa: PLC0415 - layer above
    from solprop.physdata.mixtures import Mixture  # noqa: PLC0415
    from solprop.recipe.pins import _route_pulse  # noqa: PLC0415
    from solprop.tools import (  # noqa: PLC0415
        SolitonOrder,
        _fill_n2,
        nonlinear_coefficient,
        phase_mismatch,
    )

    wavelength = keywords.get("wavelength")
    if not isinstance(wavelength, int | float) or soliton is None:
        return None, {}
    pump = float(wavelength)
    phases = keywords.get("phases")
    if phases is not None and np.any(np.asarray(phases, dtype=float)[2:] != 0.0):
        # The design numbers are an unchirped pulse's: a chirped one is linear.
        return pump, {}
    try:
        # The order is a placeholder: the target reads the pulse's numbers and
        # refuses the pulses they are not defined for.
        target = SolitonOrder(1.0, _route_pulse(keywords))  # type: ignore[arg-type]
        given = keywords.get("gamma")
        if isinstance(given, int | float) and not isinstance(given, bool):
            gamma = float(given)
        elif isinstance(mixture := getattr(soliton, "mixture", None), Mixture):
            n2, _ = _fill_n2(mixture, pump, z)
            gamma = nonlinear_coefficient(soliton, n2, pump, z=z)
        else:
            return pump, {}  # no nonlinear coefficient: the linear condition
    except KeyError, TypeError, ValueError, ToolsError:
        # No width or size, a shape without design numbers, or no Kerr data.
        return pump, {}
    carrier = np.array([2 * np.pi * _C / pump])
    if not float(beta2(soliton, carrier, z)[0]) < 0.0:
        # A pump in normal dispersion launches no soliton: the linear condition.
        return pump, {}
    pulse = {
        "peak_power": target.peak_power,
        "gamma": gamma,
        "fwhm": target.fwhm,
        "shape": target.shape,
    }
    try:
        # Below the soliton threshold there is no largest constituent.
        phase_mismatch(soliton, carrier, pump, z=z, **pulse)  # type: ignore[arg-type]
    except ToolsError:
        return pump, {}
    return pump, pulse


def _phase_matching(  # noqa: PLR0913 - a mode pair, the band, the pump and its pulse
    mode: "Mode",  # noqa: UP037 - TYPE_CHECKING only
    soliton: "Mode",  # noqa: UP037 - TYPE_CHECKING only
    omega: Float[np.ndarray, " n_omega"],
    *,
    z: float,
    pump: float | None,
    pulse: Mapping[str, object],
    limits: tuple[float, float],
) -> tuple[Float[np.ndarray, " n_omega"] | None, tuple[float, ...]]:
    """Return a mode's phase mismatch to the soliton, and its phase-matched waves.

    Parameters
    ----------
    mode : solprop.modes.base.Mode
        The mode the dispersive wave is in.
    soliton : solprop.modes.base.Mode
        The launch mode, the soliton's.
    omega : numpy.ndarray
        The band, rad/s.
    z : float
        The position, m.
    pump : float or None
        The pump wavelength, m.
    pulse : collections.abc.Mapping
        The nonlinear term's keywords (:func:`_pump`), empty for the linear one.
    limits : tuple of float
        The band's shortest and longest wavelengths, m.

    Returns
    -------
    tuple
        ``phase_mismatch`` and ``rdw``, the :class:`ModePreview` fields.
    """
    from solprop.tools import lambda_rdw_all, phase_mismatch  # noqa: PLC0415

    if pump is None:
        return None, ()
    arguments = {"soliton_mode": soliton, "z": z, **pulse}
    window = (limits[0], min(limits[1], 0.9 * pump))
    try:
        mismatch = phase_mismatch(mode, omega, pump, **arguments)  # type: ignore[arg-type]
        roots = (
            lambda_rdw_all(mode, pump, wavelength_limits=window, **arguments)  # type: ignore[arg-type]
            if window[0] < window[1]
            else ()
        )
    except ToolsError:
        # A pump outside the band the mode is defined on.
        return None, ()
    return np.asarray(mismatch), tuple(map(float, roots))


def _gamma(
    mode: "Mode",  # noqa: UP037 - TYPE_CHECKING only
    omega: Float[np.ndarray, " n_omega"],
    area: float,
    carrier: float,
    z: float,
) -> Float[np.ndarray, " n_omega"] | None:
    """Return a mode's nonlinear coefficient over the band, if its fill has one.

    Parameters
    ----------
    mode : solprop.modes.base.Mode
        The mode.
    omega : numpy.ndarray
        Angular frequencies, rad/s.
    area : float
        Its effective area, m^2.
    carrier : float
        The wavelength the fill's Kerr data are taken at, m.
    z : float
        The position, m.

    Returns
    -------
    numpy.ndarray or None
        ``gamma`` in rad/(W m); ``None`` without a fill or its Kerr data.
    """
    from solprop.physdata.mixtures import Mixture  # noqa: PLC0415
    from solprop.tools import _fill_n2  # noqa: PLC0415 - the tools' own n2

    mixture = getattr(mode, "mixture", None)
    if not isinstance(mixture, Mixture):
        return None
    try:
        n2, _ = _fill_n2(mixture, carrier, z)
    except ToolsError:
        return None
    return np.asarray(n2 * omega / (_C * area))


def preview_input(recipe: "Recipe") -> InputPreview:  # noqa: UP037 - run time
    """Return the input pulse on the run's grid, as the launch sets it, without noise.

    The recipe is built without its noise (its seed with it) and translated, which
    stops before a backend is built; the launched field is read through the
    processing functions that read a run's file. On the free-space routes the beam's
    power, energy and fluence are measured in the medium at the launch plane, as the
    tutorials measure a run's saved plane.

    Parameters
    ----------
    recipe : Recipe
        The recipe.

    Returns
    -------
    InputPreview
        The pulse in time and frequency, its energy and its peak power, and on a
        free-space route its fluence.
    """
    from solprop.driver.plane_view import PlaneView  # noqa: PLC0415 - layer D
    from solprop.output.schema import SavedPlane  # noqa: PLC0415
    from solprop.recipe.metadata import OBJECT_ROUTES, parameters  # noqa: PLC0415

    quiet = recipe.resolve()
    build = quiet._document["build"]
    if "noise" in parameters(recipe.route):
        # The planar route takes no noise keyword; it launches none.
        build["noise"] = "none"
        build.pop("seed", None)
    simulation = quiet.build()
    problem = simulation.translate()
    z = float(problem.initial.z)
    # The launched state, as a run stores its entrance.
    field = np.asarray(problem.initial.field0)
    plane = SavedPlane(z=z, field=field, stats={}, noise_phase=None)
    view = PlaneView(plane, 0, simulation.metadata(), None, ObserverSpend())
    try:
        omega, spectrum = view.spectrum()
        traced = None if recipe.route in OBJECT_ROUTES else view.power()
    finally:
        view._close()
    if traced is None:
        # A free-space beam's power is its axial flux: the view's power would read
        # the transverse columns as modes, and refuses them. The grid's coarse axis
        # is the centred one the processing functions return.
        beam = _beam(simulation, field, z)
        return InputPreview(
            time=np.asarray(simulation.grid.t),
            power=beam.power,
            omega=np.asarray(omega),
            spectrum=np.asarray(spectrum),
            energy=beam.energy,
            peak_power=float(np.max(beam.power)),
            fluence=beam.fluence,
            axes=beam.axes,
        )
    time, power = traced
    power = np.asarray(power)
    total = power.reshape(-1, power.shape[-1]).sum(axis=0) if power.ndim > 1 else power
    return InputPreview(
        time=np.asarray(time),
        power=total,
        omega=np.asarray(omega),
        spectrum=np.asarray(spectrum),
        energy=float(np.trapezoid(total, time)),
        peak_power=float(np.max(total)),
    )


@dataclass(frozen=True)
class _Beam:
    """A free-space launch measured at its plane.

    Attributes
    ----------
    power : numpy.ndarray
        The axial power through the plane, W (W/m on a line).
    energy : float
        Its time integral, J (J/m on a line).
    fluence : numpy.ndarray
        The fluence at the transverse nodes, J/m^2, shaped as :class:`InputPreview`
        says.
    axes : tuple of numpy.ndarray
        The fluence's axes, m.
    """

    power: Float[np.ndarray, " n_t"]
    energy: float
    fluence: Float[np.ndarray, "*nodes"]
    axes: tuple[Float[np.ndarray, " _axis"], ...]


def _beam(
    simulation: "Simulation",  # noqa: UP037 - run time
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    z: float,
) -> _Beam:
    """Measure a free-space launch in its medium at the launch plane.

    Parameters
    ----------
    simulation : solprop.driver.simulation.Simulation
        The built simulation, whose medium and grids measure the plane.
    field : numpy.ndarray
        The launched spectrum, internal, shape ``(n_modes, n_pol, n_omega)``.
    z : float
        The launch position, m.

    Returns
    -------
    _Beam
        Its power, energy and fluence, through the functions that measure a run's
        saved plane.
    """
    from solprop.grid import FreeGrid, LineGrid, RadialGrid  # noqa: PLC0415
    from solprop.spatial import (  # noqa: PLC0415 - the free-space routes only
        SpatialFlux,
        planar_fluence,
        spatial_energy,
        spatial_fluence,
        spatial_power,
    )

    # A free-space route's simulation carries both.
    space = cast("RadialGrid | FreeGrid | PlanarGrid", simulation.spatial_grid)
    medium = cast("SpatialMedium", simulation.medium)
    flux = SpatialFlux.resolve(simulation.grid, space, medium, z=z)
    axes: tuple[np.ndarray, ...]
    if isinstance(space, RadialGrid):
        fluence, axes = spatial_fluence(field, flux), (np.asarray(space.r),)
    elif isinstance(space, LineGrid):
        fluence, axes = spatial_fluence(field, flux), (np.asarray(space.x),)
    elif isinstance(space, FreeGrid):
        fluence = spatial_fluence(field, flux).reshape(space.n_y, space.n_x)
        axes = (np.asarray(space.y), np.asarray(space.x))
    else:
        # The planar nodes are flattened ix + n_x iy, x on the slab rule's nodes.
        planar = cast("PlanarGrid", space)
        x = np.asarray(planar.x)
        fluence = planar_fluence(field, flux).reshape(planar.n_y, x.size)
        axes = (np.asarray(planar.y), x)
    return _Beam(
        power=np.asarray(spatial_power(field, flux)),
        energy=spatial_energy(field, flux),
        fluence=np.asarray(fluence),
        axes=axes,
    )

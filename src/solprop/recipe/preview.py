# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Previews: what a recipe will build, shown before the run (SEP 41, S18).

:func:`modes` gives the mode set a guided route builds, with its dispersion, loss and
effective area over the grid's band; :func:`preview_input` gives the input pulse on the
run's grid as the launch sets it, without noise, through the same processing functions
that read a run's file, and on the free-space routes the beam's fluence across the
launch plane too. Both resolve the recipe's pins first.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop._host import ObserverSpend
from solprop.exceptions import ModeError
from solprop.recipe.errors import RecipeError

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation
    from solprop.grid import FreeGrid, RadialGrid
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

GUIDED = ("capillary", "waveguide", "planar")
"""The routes with a mode set to preview.

The planar route's mode set is its grid's slab channels in x, the modes the beam is
projected onto, at ``k_y = 0``.
"""


@dataclass(frozen=True)
class ModePreview:
    """One mode of a guided route, over the grid's band.

    Attributes
    ----------
    label : str
        Its display label, as ``launch`` and ``channel`` name it.
    beta : numpy.ndarray
        Propagation constant in 1/m.
    beta2 : numpy.ndarray
        Group-velocity dispersion in s^2/m.
    alpha : numpy.ndarray
        Power loss coefficient in 1/m.
    aeff : float
        Effective area in m^2; for a slab mode, an effective width in m.
    zdw : float or None
        Zero-dispersion wavelength in m within the band, or ``None`` where the
        dispersion keeps one sign across it (or changes it more than once).
    """

    label: str
    beta: Float[np.ndarray, " n_omega"]
    beta2: Float[np.ndarray, " n_omega"]
    alpha: Float[np.ndarray, " n_omega"]
    aeff: float
    zdw: float | None


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
    """

    z: float
    omega: Float[np.ndarray, " n_omega"]
    modes: tuple[ModePreview, ...]


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

    Its cost is the mode solve: on a guide route, the dense tables, with the memory
    admission the route's constructor makes.

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
    """
    from solprop.modes.base import aeff, alpha, beta, beta2, zdw  # noqa: PLC0415

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
    previews = []
    for mode in channels:
        try:
            zero: float | None = zdw(mode, at, wavelength_limits=limits)
        except ModeError:
            zero = None
        previews.append(
            ModePreview(
                label=mode.channel_label.display_label,
                beta=beta(mode, omega, at),
                beta2=beta2(mode, omega, at),
                alpha=alpha(mode, omega, at),
                aeff=aeff(mode, at),
                zdw=zero,
            )
        )
    return ModeSetPreview(at, omega, tuple(previews))


_C = 299_792_458.0


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
    # The launched state's one batch member, as a run stores its entrance.
    field = np.asarray(problem.initial.field0)[0]
    plane = SavedPlane(z=z, field=field, stats={}, noise_phase=None)
    view = PlaneView(plane, 0, simulation.metadata(), None, ObserverSpend())
    try:
        time, power = view.power()
        omega, spectrum = view.spectrum()
    finally:
        view._close()
    if recipe.route in OBJECT_ROUTES:
        beam = _beam(simulation, field, z)
        return InputPreview(
            time=np.asarray(time),
            power=beam.power,
            omega=np.asarray(omega),
            spectrum=np.asarray(spectrum),
            energy=beam.energy,
            peak_power=float(np.max(beam.power)),
            fluence=beam.fluence,
            axes=beam.axes,
        )
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

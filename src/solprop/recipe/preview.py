# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Previews: what a recipe will build, shown before the run (SEP 41, S18).

:func:`modes` gives the mode set a guided route builds, with its dispersion, loss and
effective area over the grid's band; :func:`preview_input` gives the input pulse on the
run's grid as the launch sets it, without noise, through the same processing functions
that read a run's file. Both resolve the recipe's pins first.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Complex, Float

from solprop._host import ObserverSpend
from solprop.exceptions import ModeError
from solprop.recipe.errors import RecipeError

if TYPE_CHECKING:
    from solprop.recipe.core import Recipe

__all__ = [
    "GUIDED",
    "InputPreview",
    "ModePreview",
    "ModeSetPreview",
    "modes",
    "preview_input",
]

GUIDED = ("capillary", "waveguide")
"""The routes with a mode set to preview."""


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
        Effective area in m^2.
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

    Attributes
    ----------
    time : numpy.ndarray
        The time axis, s.
    power : numpy.ndarray
        The instantaneous power, W, summed over modes.
    omega : numpy.ndarray
        The angular frequency axis, rad/s.
    spectrum : numpy.ndarray
        The complex spectrum, per mode, as
        :func:`~solprop.processing.fields.get_spectrum` gives it.
    energy : float
        The pulse energy, J.
    peak_power : float
        The peak power, W.
    """

    time: Float[np.ndarray, " n_t"]
    power: Float[np.ndarray, " n_t"]
    omega: Float[np.ndarray, " n_omega"]
    spectrum: Complex[np.ndarray, "*modes n_omega"]
    energy: float
    peak_power: float


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
    omega = np.sort(np.asarray(grid.omega)[grid.inband_mask])
    at = float(simulation.initial_z or 0.0) if z is None else float(z)
    limits = (
        2 * np.pi * _C / float(omega[-1]),
        2 * np.pi * _C / float(omega[0]),
    )
    previews = []
    for mode in simulation.modes:
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
    processing functions that read a run's file.

    Parameters
    ----------
    recipe : Recipe
        The recipe.

    Returns
    -------
    InputPreview
        The pulse in time and frequency, its energy and its peak power.
    """
    from solprop.driver.plane_view import PlaneView  # noqa: PLC0415 - layer D
    from solprop.output.schema import SavedPlane  # noqa: PLC0415

    quiet = recipe.resolve()
    build = quiet._document["build"]
    if "noise" in quiet._table():
        build["noise"] = "none"
        build.pop("seed", None)
    simulation = quiet.build()
    problem = simulation.translate()
    plane = SavedPlane(
        z=float(problem.initial.z),
        # The launched state's one batch member, as a run stores its entrance.
        field=np.asarray(problem.initial.field0)[0],
        stats={},
        noise_phase=None,
    )
    view = PlaneView(plane, 0, simulation.metadata(), None, ObserverSpend())
    try:
        time, power = view.power()
        omega, spectrum = view.spectrum()
    finally:
        view._close()
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

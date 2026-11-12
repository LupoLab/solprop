# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The spatial routes: free space and planar grids, and their transverse design."""

import dataclasses
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from jaxtyping import Float

from solprop.boundaries import SpatialAbsorber
from solprop.driver.defaults import resolve_noise
from solprop.driver.inputs import (
    _bind_spatial_rates,
    _check_spatial_input,
    _redrawable,
    _spatial_draw,
)
from solprop.driver.mixtures import mixture_profiles
from solprop.driver.settings import (
    NoiseSetting,
    SimulationSettings,
    _spatial_default,
    _spatial_settings,
)
from solprop.exceptions import BoundaryWarning, DriverError, TranslationError
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.linear import Frame
from solprop.modes.slab import PlanarGrid, SlabMode
from solprop.physdata import Gas, Mixture, as_gas, fused_silica, species
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.species import default_species
from solprop.profiles import ZProfile
from solprop.pulses import SpatialPulse, SpatialPulseSum
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.spatial_absorber import SpatialAbsorberDesign
from solprop.stats import default_spec
from solprop.translate.problem import PropagationProblem
from solprop.translate.responses import declares_harmonic_carrier
from solprop.translate.routes import transverse_of
from solprop.translate.transverse import checked_design, resolve_transverse

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = []


def _from_spatial(  # noqa: PLR0917 - called positionally
    cls: type["Simulation"],
    grid: TimeGrid,
    space: RadialGrid | FreeGrid,
    medium: SpatialMedium,
    responses: Sequence[object],
    initial: SpatialPulse | SpatialPulseSum | np.ndarray,
    *,
    settings: SimulationSettings,
    frame: Frame | None = None,
    noise: NoiseSetting = "default",
    seed: int | None = None,
    materials: MaterialSelection | None = None,
) -> "Simulation":
    """Build a radial or Cartesian propagation using physical fields.

    - ``space``: radial nodes, outer radius and spatial collar, or the Cartesian
      axes and their collar.
    - ``initial``: beam specification or internal spectrum, a Hankel spectrum in V m
      or a raw Cartesian DFT spectrum in V/m, with shape ``(n_columns, n_pol,
      n_omega)``.
    - ``settings``: step tolerances, positive save positions in m, statistics and
      backend. Absolute tolerance has the input spectrum's units.
    - ``frame``: moving frame. None uses the declared responses' THG choice.
    - ``seed``: random generator seed; None draws and records system entropy.
    - Returns: cached translation and the ordinary resumable execution interface.
    """
    chosen_frame = frame
    if chosen_frame is None:
        chosen_frame = Frame.comoving(
            grid, thg=declares_harmonic_carrier(responses, grid)
        )
    if isinstance(medium.material, Mixture) and settings.stats is None:
        base = default_spec("radial" if isinstance(space, RadialGrid) else "free")
        settings = dataclasses.replace(
            settings,
            stats=dataclasses.replace(
                base,
                names=(
                    *base.names,
                    "species_ionisation",
                    "species_electron_density",
                    "species_peak_ionisation_rate",
                    "species_density",
                    "species_ionisation_energy",
                ),
            ),
        )
    settings = _spatial_default(settings, grid)
    if isinstance(initial, np.ndarray):
        # Refused before the simulation copies it.
        _check_spatial_input(initial.shape, space.n_columns, grid.n_omega)
    simulation = cls(
        grid,
        (),
        responses,
        initial,
        settings=_spatial_settings(
            settings, "radial" if isinstance(space, RadialGrid) else "free"
        ),
        profiles=mixture_profiles(medium.material)
        if isinstance(medium.material, Mixture)
        else {"density": medium.density},
        spatial_grid=space,
        medium=medium,
        frame=chosen_frame,
        materials=materials,
    )
    _check_spatial_input(simulation._field0.shape, space.n_columns, grid.n_omega)
    noise_model, resolution = resolve_noise(
        "radial" if isinstance(space, RadialGrid) else "free",
        noise,
        seed,
    )
    simulation.noise_resolution = resolution
    if noise_model is not None:
        draw = _spatial_draw(noise_model, simulation, medium)
        simulation.noise = draw(seed)
        _redrawable(simulation, draw, seed)
    _bind_spatial_rates(simulation, initial)
    return simulation


def _rechecked(
    simulation: "Simulation",
    original: PropagationProblem,
    derived: PropagationProblem,
    planes: Float[np.ndarray, " n_z"],
) -> SpatialAbsorberDesign | None:
    """Return a derived run's transverse design, its check rerun on the new launch.

    SEP 40 section 5.2: a derived run keeps its parent's design, which its shared
    operator and transform were built with, but not its verdict, which belongs to a
    launch. Only a scalar multiple of the checked launch reuses it, because the check
    is linear and normalised by the launch's energy. A new launch whose rms
    wavenumber lies above the one the band was derived from warns as well.

    - ``planes``: the derived run's save planes, m.
    """
    design = transverse_of(original.route)
    if design is None or not design.has_collar:
        return design
    space = simulation.spatial_grid
    medium = simulation.medium
    assert isinstance(space, RadialGrid | FreeGrid | PlanarGrid)
    assert medium is not None
    old = np.asarray(original.initial.field0)
    new = np.asarray(derived.initial.field0)
    reference = float(np.vdot(old, old).real)
    if reference > 0.0:
        scale = np.vdot(old, new) / reference
        residual = float(np.linalg.norm(new - scale * old))
        if residual <= 1e-12 * float(np.linalg.norm(new)):
            return design
    length = float(np.max(planes)) - float(derived.initial.z)
    dispersion = SpatialDispersion(
        medium, simulation.grid, Frame.comoving(simulation.grid)
    )
    if design.specification.band_source == "launch":
        before = resolve_transverse(
            design.specification, space, dispersion, old, length
        )
        after_kappa = _launch_band(design.specification, space, dispersion, new, length)
        if after_kappa is not None and after_kappa > max(before.kappa_p) * (1 + 1e-9):
            warnings.warn(
                "the new launch spreads beyond the band its run's collar was designed "
                f"for ({after_kappa:.3g} > {max(before.kappa_p):.3g} rad/m): what it "
                "carries above that band is absorbed only partly; translate afresh",
                BoundaryWarning,
                stacklevel=3,
            )
    try:
        return checked_design(
            dataclasses.replace(design, check={}), space, dispersion, new, length
        )
    except TranslationError as error:
        msg = (
            f"{error}. The grid is shared with the parent run, so translate a new "
            "simulation with that collar_fraction for this launch"
        )
        raise DriverError(msg) from error


def _launch_band(
    specification: SpatialAbsorber,
    space: RadialGrid | FreeGrid | PlanarGrid,
    dispersion: SpatialDispersion,
    launch: np.ndarray,
    length: float,
) -> float | None:
    """Return the widest protected band a launch would derive, or ``None``.

    - ``launch``: the launch, canonical layout.
    - ``length``: the run's length, m.
    - Returns: the widest axis's band edge, rad/m; ``None`` when it cannot be resolved.
    """
    try:
        return max(
            resolve_transverse(specification, space, dispersion, launch, length).kappa_p
        )
    except TranslationError:
        return None


def _refuse_pending_check(problem: PropagationProblem) -> None:
    """Refuse to start a run whose collar's design check never saw a launch.

    A coupled child is translated before its launch exists; its check runs when
    :meth:`Simulation.with_input` gives it one (SEP 40 section 5.2).
    """
    design = transverse_of(problem.route)
    if design is not None and design.check.get("status") == "pending":
        msg = (
            "this run's collar has not been checked against a launch: give it one "
            "with with_input before running it (SEP 40 section 5.2)"
        )
        raise DriverError(msg)


def _selected_medium(
    medium: SpatialMedium, materials: MaterialSelection
) -> SpatialMedium:
    """Resolve a bulk medium's built-in gas under a material selection.

    A medium that names a built-in gas meant the built-in gas: it becomes that gas's
    one-population mixture with the selection's models, at the same pressure path and
    temperature, so that it is recorded; a supplied susceptibility is kept beside
    optmat's equation of state, as for a capillary fill. Any other medium is taken as
    it is.
    """
    if not isinstance(medium.material, Gas):
        return medium
    gas = medium.material
    selected = species(gas, selection=materials)
    if medium.susceptibility != default_species(gas).susceptibility:
        selected = dataclasses.replace(selected, susceptibility=medium.susceptibility)
    # A constructed medium holds its resolved temperature, a number.
    fill = Mixture.fill(
        {selected: 1.0}, medium.pressure, temperature=cast("float", medium.temperature)
    )
    return SpatialMedium(fill)


def _selected_planar(space: PlanarGrid, materials: MaterialSelection) -> PlanarGrid:
    """Resolve a planar grid's built-in materials under a material selection.

    The slab modes' fill of a built-in gas with its default susceptibility meant the
    built-in gas: it takes the selection's single-species susceptibility
    (:func:`solprop.physdata.species.species`), and keeps its pressure, temperature
    and optmat density path, so no mixture -- which the hybrid transform does not
    admit -- is needed (SEP 3, Q5). A slab declared without a cladding takes optmat's
    recommended silica. A supplied susceptibility and a declared cladding are the
    caller's.

    - Returns: the grid given when nothing resolves; otherwise the same grid rebuilt on
      the resolved slab modes.
    """
    modes = cast("tuple[SlabMode, ...]", tuple(space.modes))
    first = modes[0]
    changes: dict[str, object] = {}
    if first.mixture is None and first.gas is not None:
        gas = as_gas(first.gas)
        if first.susceptibility == default_species(gas).susceptibility:
            # The modes share one fill (`PlanarGrid` refuses any other), so one
            # resolved susceptibility serves them all and the medium is checked
            # against them by identity.
            resolved = species(gas, selection=materials).susceptibility
            if resolved != first.susceptibility:
                changes["susceptibility"] = resolved
    cladding = fused_silica()
    rebuilt = tuple(
        dataclasses.replace(
            mode,
            **changes,
            **({"cladding": cladding} if mode.cladding is None else {}),
        )
        if changes or mode.cladding is None
        else mode
        for mode in modes
    )
    if all(one is two for one, two in zip(rebuilt, modes, strict=True)):
        return space
    with warnings.catch_warnings():
        # The caller's own grid already said so if its n_x is below the bound.
        warnings.filterwarnings(
            "ignore", message=r"PlanarGrid n_x = ", category=UserWarning
        )
        return dataclasses.replace(space, modes=rebuilt)


def _planar_medium(space: PlanarGrid) -> SpatialMedium:
    """Return the fill of a planar grid's modes as a constant bulk medium."""
    mode = cast("SlabMode", space.modes[0])
    if mode.mixture is not None:
        msg = (
            "a planar run takes a single-gas fill: a mixture on the hybrid planar "
            "transform is not admitted yet; fill the slab with gas= and pressure="
        )
        raise DriverError(msg)
    if not mode.density.is_constant:
        msg = (
            "the hybrid planar transform requires constant media: a gas path along z "
            "(a varying pressure or temperature) is refused, as for Cartesian free "
            "space, whose seam evaluates a constant operator exactly and has no "
            "varying-medium expansion"
        )
        raise DriverError(msg)
    if mode.gas is None:
        return SpatialMedium()
    return SpatialMedium(
        mode.gas,
        cast("float | ZProfile", mode.pressure),
        temperature=cast("float", mode.temperature),
        susceptibility=mode.susceptibility,
    )


def _transverse_specification(
    setting: SpatialAbsorber | Literal[False] | None,
    spatial_grid: object,
) -> SpatialAbsorber | None:
    """Return the transverse boundary a run's translation resolves (SEP 40).

    - ``spatial_grid``: the run's transverse grid, ``None`` for a modal run.
    - Returns: the specification; ``None`` for no ramp and no collar.
    """
    if spatial_grid is None:
        if isinstance(setting, SpatialAbsorber):
            msg = (
                "spatial_absorber sets a free-space run's transverse boundary; a "
                "modal route has no transverse window"
            )
            raise DriverError(msg)
        return None
    if setting is False:
        return None
    return SpatialAbsorber(real_space=False) if setting is None else setting

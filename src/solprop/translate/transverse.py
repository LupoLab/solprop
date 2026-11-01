# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resolve a free-space run's transverse boundary at translation (SEP 40 section 5.2).

:func:`resolve_transverse` turns a :class:`~solprop.spatial_absorber.SpatialAbsorber`
into a :class:`~solprop.spatial_absorber.SpatialAbsorberDesign` against the grid, the
medium over the requested band and along the run, and the noise-free launch: the
safety ramp's rate on the spectral columns and, when a collar is asked for, its rate
on the nodes, its band, its strength and its step limit. The formulas are
:mod:`solprop.spatial_absorber`'s; this module only gathers what they are evaluated on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float

from solprop.constants import C
from solprop.exceptions import GridError, TranslationError
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.modes.slab import PlanarGrid
from solprop.spatial import SpatialDispersion
from solprop.spatial_absorber import (
    CROSSING_ADAPTIVE,
    LAUNCH_BAND_FACTOR,
    PROFILE_FLOOR,
    PROTECTED_CUTOFF,
    SpatialAbsorber,
    SpatialAbsorberDesign,
    collar_attenuation,
    collar_strength,
    crossing_limit,
    reflection_threshold,
    rms_wavenumber,
    safety_ramp_strength,
)

__all__ = ["band_wavenumbers", "resolve_transverse"]


@dataclass(frozen=True)
class _Axis:
    """One transverse axis a face is normal to.

    Attributes
    ----------
    name : str
        ``"r"``, ``"x"`` or ``"y"``.
    columns : numpy.ndarray
        Each spectral column's wavenumber along the axis, rad/m (``|k_perp|``
        radially).
    size : float
        The requested half-size along it, m.
    radial : bool
        Whether the wavenumber is ``|k_perp|`` on the radial grid.
    """

    name: str
    columns: Float[np.ndarray, " n_columns"]
    size: float
    radial: bool = False


def _axes(space: RadialGrid | FreeGrid | PlanarGrid) -> tuple[_Axis, ...]:
    """Return the axes whose faces bound `space`, with each column's wavenumber.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The transverse grid.

    Returns
    -------
    tuple of _Axis
        One radially; x and y on a Cartesian grid, x alone on a line; y on a planar
        grid, whose columns are ``p + n_p * iy``.
    """
    if isinstance(space, RadialGrid):
        return (_Axis("r", np.asarray(space.k_r), space.radius, radial=True),)
    if isinstance(space, PlanarGrid):
        return (_Axis("y", np.repeat(space.k_y, space.n_p), space.radius_y),)
    x = _Axis("x", np.tile(space.k_x, space.n_y), space.radius_x)
    if isinstance(space, LineGrid):
        return (x,)
    return (x, _Axis("y", np.repeat(space.k_y, space.n_x), space.radius_y))


def _nodes_window(space: RadialGrid | FreeGrid | PlanarGrid) -> np.ndarray:
    """Return the grid's window on the nodes the collar's rate is applied at.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The transverse grid.

    Returns
    -------
    numpy.ndarray
        One inside the requested size, falling to zero at the outer size.
    """
    if isinstance(space, PlanarGrid):
        return np.repeat(space.window_y, int(space.x.size))
    return np.asarray(space.window)


def band_wavenumbers(
    space: RadialGrid | FreeGrid | PlanarGrid,
    dispersion: SpatialDispersion,
    z_max: float,
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
    """Return the medium's wavenumbers over the requested band and along the run.

    ``k = n omega / c`` over ``inband_mask & active_mask``, at the medium's path
    nodes (the entrance, the end and every breakpoint between), so that the entrance
    alone does not set the extremes (SEP 34's varying radial medium). On a planar
    grid ``k`` is each slab mode's ``beta_p`` (SEP 40 section 4.4).

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The transverse grid.
    dispersion : solprop.spatial.SpatialDispersion
        The medium and the temporal grid.
    z_max : float
        The run's length, m.

    Returns
    -------
    k : numpy.ndarray
        Wavenumbers, rad/m, one per (node, mode, frequency).
    omega : numpy.ndarray
        The angular frequency of each, rad/s.

    Raises
    ------
    TranslationError
        If the requested band holds no active bin.
    """
    grid = dispersion.grid
    mask = grid.inband_mask & grid.active_mask
    omega = np.asarray(grid.omega[mask], dtype=np.float64)
    if omega.size == 0:
        msg = "the requested band holds no active frequency"
        raise TranslationError(msg)
    if isinstance(space, PlanarGrid):
        rows = [
            np.real(np.asarray(mode.neff(omega))) * omega / C for mode in space.modes
        ]
        return np.concatenate(rows), np.tile(omega, len(rows))
    nodes = {0.0, float(z_max)}
    for path in dispersion.medium.parameter_paths.values():
        nodes.update(float(z) for z in path.breakpoints if 0.0 <= z <= z_max)
    rows = [
        np.real(np.asarray(dispersion.medium.index(omega, z))) * omega / C
        for z in sorted(nodes)
    ]
    return np.concatenate(rows), np.tile(omega, len(rows))


def _column_power(
    space: RadialGrid | FreeGrid | PlanarGrid,
    field0: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    mask: np.ndarray,
) -> Float[np.ndarray, " n_columns"]:
    """Return each spectral column's launch energy over the requested band.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The transverse grid, for its spectral weights.
    field0 : numpy.ndarray
        The noise-free launch, canonical layout.
    mask : numpy.ndarray
        The requested band's bins.

    Returns
    -------
    numpy.ndarray
        Energy per column in the field's own units.
    """
    density = np.sum(np.abs(field0[..., mask]) ** 2, axis=(0, 2, 3))
    return np.asarray(space.spectral_weights) * density


def _ramp_profile(
    columns: Float[np.ndarray, " n_columns"], start: float, top: float
) -> Float[np.ndarray, " n_columns"]:
    """Return the raised-cosine ramp from `start` to `top` on the columns.

    Parameters
    ----------
    columns : numpy.ndarray
        Each column's ``|k|``, rad/m.
    start, top : float
        Where the ramp leaves zero and reaches one, rad/m.

    Returns
    -------
    numpy.ndarray
        The profile, in [0, 1].
    """
    if not top > start:
        return np.zeros(columns.size)
    inside = np.clip((np.abs(columns) - start) / (top - start), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * inside))


def resolve_transverse(  # noqa: PLR0912, PLR0915 - one resolution, in order
    specification: SpatialAbsorber,
    space: RadialGrid | FreeGrid | PlanarGrid,
    dispersion: SpatialDispersion,
    field0: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    z_max: float,
) -> SpatialAbsorberDesign:
    """Resolve a specification into the run's immutable design (SEP 40 section 5.2).

    The band comes from an explicit `kappa`, then an `angle`, then the launch: per
    axis ``c_b sqrt(2) kappa_rms`` over the requested band's bins, radially
    ``c_b kappa_rms`` on ``|k_perp|``, read from the noise-free launch. The safety
    ramp covers the outer `safety_fraction` of each axis's wavenumbers, or starts at
    a declared `discard_above` if that is lower, with the strength of section 4.4 and
    the larger of the axes' strengths on the maximum of their profiles. A collar's
    strength is the largest of its faces' roots.

    Parameters
    ----------
    specification : solprop.spatial_absorber.SpatialAbsorber
        What the run asked for.
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The transverse grid.
    dispersion : solprop.spatial.SpatialDispersion
        The medium and the temporal grid.
    field0 : numpy.ndarray
        The noise-free launch in the canonical layout, spectral columns.
    z_max : float
        The run's length, m.

    Returns
    -------
    solprop.spatial_absorber.SpatialAbsorberDesign
        The design, with the design check still to run (``check`` empty).

    Raises
    ------
    TranslationError
        Naming the reason, when a requested collar has no room, its band is below
        the profile's floor, reaches cutoff, lies above the ramp, or cannot come from
        a launch that is zero everywhere.
    """
    grid = dispersion.grid
    k, omega = band_wavenumbers(space, dispersion, z_max)
    mask = grid.inband_mask & grid.active_mask
    power = _column_power(space, field0, mask)
    attenuation = specification.attenuation
    axes = _axes(space)
    collar = specification.real_space
    fraction = float(space.collar_fraction)
    if collar and fraction == 0.0:
        msg = (
            "a requested real-space collar needs room outside the requested size: "
            "build the grid with collar_fraction > 0 (the guide's starting points are "
            "1.0 radially and 0.5 on the Cartesian, line and planar grids), or ask "
            "for the safety ramp alone with SpatialAbsorber(real_space=False)"
        )
        raise TranslationError(msg)
    widths = tuple(fraction * axis.size for axis in axes) if collar else ()

    # The protected band per axis: one value per band wavenumber, so that an angle's
    # band follows its frequency.
    kappa_p: list[np.ndarray] = []
    if collar:
        source = specification.band_source
        for axis in axes:
            if source == "kappa":
                assert specification.kappa is not None
                edge = np.full(k.size, specification.kappa)
            elif source == "angle":
                assert specification.angle is not None
                edge = omega * math.sin(specification.angle) / C
            else:
                rms = rms_wavenumber(power, axis.columns)
                if not rms > 0.0:
                    msg = (
                        "a requested collar takes its band from the launch, which is "
                        "zero everywhere here: declare the band with kappa or angle"
                    )
                    raise TranslationError(msg)
                factor = LAUNCH_BAND_FACTOR * (1.0 if axis.radial else math.sqrt(2.0))
                edge = np.full(k.size, factor * rms)
            kappa_p.append(edge)

    # The safety ramp, per axis.
    starts: list[float] = []
    tops: list[float] = []
    strengths: list[float] = []
    for index, axis in enumerate(axes):
        top = float(np.max(np.abs(axis.columns)))
        start = (1.0 - specification.safety_fraction) * top
        if specification.discard_above is not None:
            start = min(start, specification.discard_above)
        starts.append(start)
        tops.append(top)
        length = widths[index] if collar else axis.size
        strengths.append(
            safety_ramp_strength(0.5 * (start + top), k, length, attenuation)
            if top > start
            else 0.0
        )
    profile = np.zeros(axes[0].columns.size)
    for axis, start, top in zip(axes, starts, tops, strict=True):
        profile = np.maximum(profile, _ramp_profile(axis.columns, start, top))
    ramp_strength = max(strengths)
    ramp_rate = ramp_strength * profile
    total = float(np.sum(power))
    notes: dict[str, object] = {
        "band_k": (float(np.min(k)), float(np.max(k))),
        "ramp_strengths": tuple(strengths),
        "ramp_band_share": float(np.sum(power[profile > 0.0])) / total
        if total > 0.0
        else 0.0,
    }
    if not collar:
        return SpatialAbsorberDesign(
            specification,
            ramp_rate,
            ramp_strength,
            tuple(starts),
            notes=notes,
        )

    # The collar: refusals, then the strongest face's root.
    for axis, edge, width, start in zip(axes, kappa_p, widths, starts, strict=True):
        reddest = float(np.min(edge))
        if reddest * width < PROFILE_FLOOR:
            need = PROFILE_FLOOR / (reddest * axis.size)
            msg = (
                f"the collar on the {axis.name} faces is too narrow for its band: "
                f"kappa_p w = {reddest * width:.3g} is below the Planck profile's "
                f"floor of {PROFILE_FLOOR}; a collar_fraction of at least {need:.3g} "
                "meets it"
            )
            raise TranslationError(msg)
        if np.any(edge > PROTECTED_CUTOFF * k):
            msg = (
                f"the protected band on the {axis.name} faces reaches beyond "
                f"{PROTECTED_CUTOFF} k at some requested frequency, where the collar "
                "cannot be designed: protect a narrower band"
            )
            raise TranslationError(msg)
        if float(np.max(edge)) > start:
            msg = (
                f"the protected band on the {axis.name} faces reaches the safety "
                f"ramp, which starts at {start:.4g} rad/m: refine the grid, or protect "
                "a narrower band"
            )
            raise TranslationError(msg)
        if (
            specification.discard_above is not None
            and specification.discard_above < float(np.max(edge))
        ):
            msg = "discard_above lies below the protected band"
            raise TranslationError(msg)
    try:
        alpha0 = max(
            collar_strength(edge, k, width, attenuation)
            for edge, width in zip(kappa_p, widths, strict=True)
        )
    except GridError as error:
        raise TranslationError(str(error)) from error
    window = _nodes_window(space)
    collar_rate = alpha0 * (1.0 - window)
    kappa_lo = max(reflection_threshold(alpha0, width, k) for width in widths)
    crossing = min(
        crossing_limit(edge, k, width, CROSSING_ADAPTIVE)
        for edge, width in zip(kappa_p, widths, strict=True)
    )
    edges = tuple(float(np.min(edge)) for edge in kappa_p)
    # What would come back from the intermediate band, kappa_p < |k| < kappa_s,
    # weighted by the launch's spectrum: a report, judged by the monitor (4.4).
    intermediate = 0.0
    for axis, edge, width, start in zip(axes, kappa_p, widths, starts, strict=True):
        between = (np.abs(axis.columns) > float(np.max(edge))) & (
            np.abs(axis.columns) < start
        )
        if total > 0.0 and np.any(between):
            returned = collar_attenuation(
                np.abs(axis.columns[between]),
                np.full(int(np.count_nonzero(between)), float(np.min(k))),
                alpha0,
                width,
            )
            intermediate += float(np.sum(power[between] * np.exp(-2.0 * returned)))
    notes |= {
        "kappa_p_band": tuple(
            (float(np.min(edge)), float(np.max(edge))) for edge in kappa_p
        ),
        "intermediate_return": intermediate / total if total > 0.0 else 0.0,
    }
    if isinstance(space, RadialGrid):
        notes["reflected_columns"] = int(np.count_nonzero(space.k_r < kappa_lo))
    return SpatialAbsorberDesign(
        specification,
        ramp_rate,
        ramp_strength,
        tuple(starts),
        collar_rate=collar_rate,
        alpha0=alpha0,
        width=widths,
        kappa_p=edges,
        kappa_lo=kappa_lo,
        crossing_step=crossing,
        unguarded_above=(0.75 / CROSSING_ADAPTIVE) * min(edges),
        notes=notes,
    )

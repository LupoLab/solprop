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

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Float
from scipy.special import j0

from solprop.constants import C
from solprop.exceptions import GridError, TranslationError
from solprop.grid import FreeGrid, LineGrid, RadialGrid, TimeGrid
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
from solprop.translate.collar_check import BUDGET, design_check

__all__ = ["LADDER", "band_wavenumbers", "checked_design", "resolve_transverse"]


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


def _intermediate_return(  # noqa: PLR0913, PLR0917 - the design and its weights
    axes: tuple[_Axis, ...],
    kappa_p: list[np.ndarray],
    widths: tuple[float, ...],
    starts: list[float],
    alpha0: float,
    field0: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    space: RadialGrid | FreeGrid | PlanarGrid,
    grid: TimeGrid,
    k: Float[np.ndarray, " n"],
) -> float:
    """Return the launch's energy share the intermediate band would bring back.

    SEP 40 section 4.4: content between the protected band and the ramp is absorbed
    at the edge only to ``Gamma(kappa)``; its return is ``exp(-2 Gamma)`` with the
    exact symbol per frequency, weighted by the launch's spectrum. The band is taken in
    sixteen contiguous groups of bins, each at its power-weighted wavenumber.

    Parameters
    ----------
    axes : tuple of _Axis
        The grid's axes.
    kappa_p : list of numpy.ndarray
        The protected band's edge per axis.
    widths : tuple of float
        The collar's width per axis, m.
    starts : list of float
        The ramp's start per axis, rad/m.
    alpha0 : float
        The collar's peak rate, 1/m.
    field0 : numpy.ndarray
        The noise-free launch.
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The grid, for its spectral weights.
    grid : solprop.grid.TimeGrid
        The temporal grid.
    k : numpy.ndarray
        The band's wavenumbers, the entrance's first.

    Returns
    -------
    float
        A share of the launch energy.
    """
    band = np.nonzero(grid.inband_mask & grid.active_mask)[0]
    power = np.asarray(space.spectral_weights)[:, None] * np.sum(
        np.abs(np.asarray(field0[0])[..., band]) ** 2, axis=1
    )
    total = float(np.sum(power))
    if not total > 0.0:
        return 0.0
    entrance = np.asarray(k[: band.size])
    groups = np.array_split(np.arange(band.size), min(128, band.size))
    spectrum = np.sum(power, axis=0)
    group_k = np.array(
        [
            float(np.average(entrance[g], weights=spectrum[g]))
            if float(np.sum(spectrum[g])) > 0.0
            else float(np.mean(entrance[g]))
            for g in groups
        ]
    )
    group_power = np.stack([np.sum(power[:, g], axis=1) for g in groups], axis=1)
    returned = 0.0
    for axis, edge, width, start in zip(axes, kappa_p, widths, starts, strict=True):
        magnitude = np.abs(axis.columns)
        between = (magnitude > float(np.max(edge))) & (magnitude < start)
        if not np.any(between):
            continue
        values, inverse = np.unique(magnitude[between], return_inverse=True)
        loss = np.empty((values.size, group_k.size))
        for chunk in range(0, values.size, 16):
            part = values[chunk : chunk + 16]
            loss[chunk : chunk + 16] = collar_attenuation(
                part[:, None], group_k[None, :], alpha0, width
            )
        returned += float(np.sum(group_power[between] * np.exp(-2.0 * loss[inverse])))
    return returned / total


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
    intermediate = _intermediate_return(
        axes, kappa_p, widths, starts, alpha0, field0, space, grid, k
    )
    notes |= {
        "kappa_p_band": tuple(
            (float(np.min(edge)), float(np.max(edge))) for edge in kappa_p
        ),
        "intermediate_return": intermediate,
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


LADDER = (0.5, 1.0, 1.5, 2.0, 3.0)
"""The collar fractions the design check tries, in order, to name one that passes."""


def _trial_grid(
    space: RadialGrid | FreeGrid | PlanarGrid, fraction: float
) -> RadialGrid | FreeGrid | PlanarGrid:
    """Return `space` with another collar fraction at the same spacing.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The design's grid.
    fraction : float
        The trial fraction.

    Returns
    -------
    solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The trial grid.
    """
    scale = (1.0 + fraction) / (1.0 + float(space.collar_fraction))
    if isinstance(space, RadialGrid):
        return RadialGrid(space.radius, max(2, round(space.n_r * scale)), fraction)
    if isinstance(space, PlanarGrid):
        return dataclasses.replace(
            space, n_y=max(2, round(space.n_y * scale)), collar_fraction=fraction
        )
    if isinstance(space, LineGrid):
        return LineGrid(space.radius_x, max(2, round(space.n_x * scale)), fraction)
    return FreeGrid(
        space.radius_x,
        max(2, round(space.n_x * scale)),
        space.radius_y,
        max(2, round(space.n_y * scale)),
        fraction,
    )


def _relaunched(
    space: RadialGrid | FreeGrid | PlanarGrid,
    trial: RadialGrid | FreeGrid | PlanarGrid,
    field0: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
) -> Complex[np.ndarray, "1 n_trial n_pol n_omega"]:
    """Return the launch on a trial grid: its field, zero beyond the design's grid.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The design's grid.
    trial : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The trial grid, the same spacing.
    field0 : numpy.ndarray
        The launch's spectral columns on `space`.

    Returns
    -------
    numpy.ndarray
        Its spectral columns on `trial`.
    """
    first = np.asarray(field0[:1], dtype=np.complex128)
    _, n_cols, n_pol, n_omega = first.shape
    flat = first[0].reshape(n_cols, n_pol * n_omega)
    if isinstance(space, RadialGrid):
        assert isinstance(trial, RadialGrid)
        series = j0(np.outer(trial.r, space.k_r)) * (
            space.spectral_weights / (2 * np.pi)
        )
        physical = series @ flat
        physical[np.asarray(trial.r) > space.outer_radius] = 0.0
        hankel = trial.hankel
        forward = (
            hankel.forward_scale
            * (hankel.j1[:, None] / hankel.j1[None, :])
            * hankel.matrix
        )
        columns = forward @ physical
    else:
        if isinstance(space, PlanarGrid):
            assert isinstance(trial, PlanarGrid)
            shape, new = (space.n_y, space.n_p), (trial.n_y, trial.n_p)
            axes: tuple[int, ...] = (0,)
        else:
            assert isinstance(trial, FreeGrid)
            shape, new = (space.n_y, space.n_x), (trial.n_y, trial.n_x)
            axes = (0, 1)
        cube = np.fft.ifftn(flat.reshape(*shape, -1), axes=axes)
        bigger = np.zeros((*new, cube.shape[-1]), dtype=np.complex128)
        index = tuple(
            slice((b - s) // 2, (b - s) // 2 + s) if i in axes else slice(None)
            for i, (b, s) in enumerate(zip(new, shape, strict=True))
        )
        centred = np.fft.fftshift(cube, axes=axes)
        bigger[index] = centred
        columns = np.fft.fftn(np.fft.ifftshift(bigger, axes=axes), axes=axes).reshape(
            -1, n_pol * n_omega
        )
    return columns.reshape(1, -1, n_pol, n_omega)


def checked_design(
    design: SpatialAbsorberDesign,
    space: RadialGrid | FreeGrid | PlanarGrid,
    dispersion: SpatialDispersion,
    field0: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    z_max: float,
) -> SpatialAbsorberDesign:
    """Run the design check on a collar's design and refuse a failing one by name.

    SEP 40 section 4.4. Without a collar the check is neither built nor run. A launch
    that is zero everywhere leaves it ``"pending"``, for the launch a coupled child is
    given (:meth:`solprop.driver.simulation.Simulation.with_input`).

    Parameters
    ----------
    design : solprop.spatial_absorber.SpatialAbsorberDesign
        The resolved design.
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        Its grid.
    dispersion : solprop.spatial.SpatialDispersion
        The medium and the temporal grid.
    field0 : numpy.ndarray
        The noise-free launch, canonical layout.
    z_max : float
        The run's length, m.

    Returns
    -------
    solprop.spatial_absorber.SpatialAbsorberDesign
        The design, with its check's record.

    Raises
    ------
    TranslationError
        If the check refuses the design or cannot converge, naming the first trial
        fraction that passes, if any, and whether a bare wall returns less.
    """
    if not design.has_collar:
        return design
    assert design.collar_rate is not None
    result = design_check(
        space, design.collar_rate, field0, dispersion, z_max, min(design.width)
    )
    checked = dataclasses.replace(design, check=result.record())
    if result.status in ("passed", "pending"):
        return checked
    passing = None
    for fraction in LADDER:
        if fraction <= float(space.collar_fraction):
            continue
        trial = _trial_grid(space, fraction)
        launch = _relaunched(space, trial, field0)
        try:
            candidate = resolve_transverse(
                design.specification, trial, dispersion, launch, z_max
            )
        except TranslationError:
            continue
        assert candidate.collar_rate is not None
        attempt = design_check(
            trial,
            candidate.collar_rate,
            launch,
            dispersion,
            z_max,
            min(candidate.width),
        )
        if attempt.status == "passed":
            passing = fraction
            break
    what = (
        f"is inconclusive ({result.reason})"
        if result.status == "inconclusive"
        else f"returns {result.returned:.2e} of the launch energy, above the "
        f"{BUDGET:.0e} budget"
    )
    tried = [f for f in LADDER if f > float(space.collar_fraction)]
    advice = (
        f"collar_fraction={passing:g} at the same spacing passes"
        if passing is not None
        else f"none of the trial fractions {', '.join(f'{f:g}' for f in tried)} passes"
        if tried
        else "no trial fraction above this grid's is tried; refine the grid or "
        "protect a narrower band"
    )
    wall = (
        f"; a bare wall at the same outer size returns {result.bare_wall:.2e}, "
        "less: for a slow beam, room alone may be better than a collar"
        if result.bare_wall < result.returned
        else ""
    )
    msg = (
        f"the requested collar's design check {what} (SEP 40 section 4.4): {advice}"
        f"{wall}"
    )
    raise TranslationError(msg)

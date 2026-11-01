# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The design check of a requested collar: what it sends back (SEP 40, 4.4).

A product of plane-wave reflection and a ray share cannot say how much of a launch a
collar returns: evaluated on the phase-0 witnesses it was wrong by up to nine decades,
because the return lives in diffraction tails. So the check computes it. The launch is
propagated linearly on the design's own operator, one frequency at a time, and compared
over the requested aperture with the same launch on a bare grid large enough that the
launch does not reach its edge within the run:

.. math::

    U_\mathrm{ret} = \max_{z \le L} \frac{\sum_j s_j \,
    \lVert E_\mathrm{collar}(z, \omega_j) - E_\mathrm{large}(z, \omega_j)
    \rVert^2_\mathrm{aperture}}{\sum_j s_j\, U_\mathrm{launch}(\omega_j)},

with nodes :math:`\omega_j` at equal-energy quantiles of the launch's spectrum and
shares :math:`s_j`. Radially the generator
:math:`\mathrm{diag}(i k_z) + F\,\mathrm{diag}(-\alpha/2)\,B` is exponentiated by its
eigendecomposition; on the Cartesian, line and planar grids a one-frequency Strang
march, diffraction exact in the transverse wavenumber and the collar in real space,
steps no further than the plane spacing and the crossing limit. The bare grid is
diagonal and needs no march.

The verdict is the contract of section 4.4: above :data:`BUDGET` the design is refused;
an estimate within a factor of two of it is refined once, doubling nodes, planes and
steps and enlarging the bare grid by half, and passes at :data:`REFINED_BUDGET` if the
two agree within a quarter; a check that cannot converge is inconclusive, which is not
a pass.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Complex, Float
from scipy.special import j0

from solprop.constants import C
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.hankel import QDHT
from solprop.modes.slab import PlanarGrid
from solprop.spatial import SpatialDispersion

__all__ = [
    "BUDGET",
    "REFINED_BUDGET",
    "CheckResult",
    "design_check",
]

BUDGET = 4e-8
"""The reflection budget in energy, ``(2e-4)^2`` (SEP 40 sections 4.8 and 7.1)."""

REFINED_BUDGET = 3e-8
"""What a refined estimate must meet: the budget less the 25 % by which the check
read low against full runs (Part II)."""

WARNING_SHARE = 1e-6
"""The monitor's warning share of the launch energy, which sets the launch's extent and
reach (SEP 40 sections 4.4 and 5.3)."""

_NODES = 5
_PLANES = 64
_LARGEST = 8.0
"""The bare grid may hold at most this many times the design's samples per axis."""

Status = Literal["passed", "refused", "inconclusive", "pending"]


@dataclass(frozen=True)
class CheckResult:
    """The design check's verdict and its record (SEP 40 section 4.4).

    Attributes
    ----------
    status : {'passed', 'refused', 'inconclusive', 'pending'}
        The verdict; ``"pending"`` when there is no launch to check yet.
    returned : float
        ``U_ret``, an energy share of the launch.
    bare_wall : float
        The same launch's return from a bare wall at the same outer size.
    refined : bool
        Whether the estimate was refined.
    nodes : tuple of float
        The frequency nodes, rad/s.
    planes : int
        The planes compared.
    wall_time : float
        The check's own wall time, s.
    reason : str
        Why a check did not pass or converge; empty otherwise.
    """

    status: Status
    returned: float = math.nan
    bare_wall: float = math.nan
    refined: bool = False
    nodes: tuple[float, ...] = ()
    planes: int = 0
    wall_time: float = 0.0
    reason: str = ""

    def record(self) -> dict[str, object]:
        """Return the check's record for the design.

        Returns
        -------
        dict
            JSON-ready values.
        """
        return {
            "status": self.status,
            "returned": None if math.isnan(self.returned) else self.returned,
            "bare_wall": None if math.isnan(self.bare_wall) else self.bare_wall,
            "refined": self.refined,
            "nodes": list(self.nodes),
            "planes": self.planes,
            "wall_time": self.wall_time,
            "reason": self.reason,
        }


def _frequency_nodes(
    power: Float[np.ndarray, " n_band"], count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the band bins at equal-energy quantiles of the launch, and their shares.

    Parameters
    ----------
    power : numpy.ndarray
        The launch's energy per requested-band bin.
    count : int
        How many nodes.

    Returns
    -------
    tuple of numpy.ndarray
        Indices into the band, unique, and the share each stands for.
    """
    total = float(np.sum(power))
    cumulative = np.cumsum(power) / total
    targets = (np.arange(count) + 0.5) / count
    indices = np.unique(np.searchsorted(cumulative, targets))
    indices = np.minimum(indices, power.size - 1)
    edges = np.concatenate(
        ([0.0], 0.5 * (cumulative[indices[:-1]] + cumulative[indices[1:]]), [1.0])
    )
    return indices, np.diff(edges)


def _reach(
    kappa: Float[np.ndarray, " n_columns"],
    power: Float[np.ndarray, " n_columns"],
    k: float,
) -> float:
    """Return the launch's reach: the group angle of its warning-share wavenumber.

    ``kappa_q / max(k_z(kappa_q), 0.44 k)``, with ``kappa_q`` the wavenumber above which
    the launch holds no more than :data:`WARNING_SHARE` of its energy (section 5.3).

    Parameters
    ----------
    kappa : numpy.ndarray
        Each column's ``|kappa|``, rad/m.
    power : numpy.ndarray
        Each column's energy at this frequency.
    k : float
        The medium's wavenumber, rad/m.

    Returns
    -------
    float
        ``dx/dz`` of the fastest significant content.
    """
    order = np.argsort(kappa)
    tail = np.cumsum(power[order][::-1])[::-1] / max(float(np.sum(power)), 1e-300)
    above = np.nonzero(tail > WARNING_SHARE)[0]
    kappa_q = float(kappa[order][above[-1]]) if above.size else 0.0
    k_z = math.sqrt(max(k * k - kappa_q * kappa_q, 0.0))
    return kappa_q / max(k_z, 0.44 * k)


def _extent(
    position: Float[np.ndarray, " n_nodes"], density: Float[np.ndarray, " n_nodes"]
) -> float:
    """Return the launch's extent: the radius holding all but the warning share.

    Parameters
    ----------
    position : numpy.ndarray
        Each node's distance from the axis or centre, m.
    density : numpy.ndarray
        Each node's weighted energy.

    Returns
    -------
    float
        The extent, m.
    """
    order = np.argsort(position)
    tail = np.cumsum(density[order][::-1])[::-1] / max(float(np.sum(density)), 1e-300)
    above = np.nonzero(tail > WARNING_SHARE)[0]
    return float(position[order][above[-1]]) if above.size else 0.0


def _radial_return(
    space: RadialGrid,
    rate: Float[np.ndarray, " n_r"],
    launch: Complex[np.ndarray, " n_r"],
    k: float,
    planes: np.ndarray,
    half_size: float,
) -> tuple[np.ndarray, float]:
    """Return one frequency's aperture difference energy at each plane, radially.

    Parameters
    ----------
    space : solprop.grid.RadialGrid
        The design's grid.
    rate : numpy.ndarray
        The collar's power rate on its nodes, 1/m; zeros for a bare wall.
    launch : numpy.ndarray
        The launch's spectral columns at this frequency.
    k : float
        The medium's wavenumber, rad/m.
    planes : numpy.ndarray
        Positions, m.
    half_size : float
        The bare grid's outer radius, m.

    Returns
    -------
    tuple
        The difference energy per plane, and the launch energy, in one unit.
    """
    hankel = space.hankel
    diagonal = hankel.j1[:, None] / hankel.j1[None, :]
    forward = hankel.forward_scale * diagonal * hankel.matrix
    inverse = hankel.inverse_scale * diagonal * hankel.matrix
    k_r = np.asarray(space.k_r)
    phase = np.sqrt(k * k - k_r * k_r + 0j) - k
    generator = np.diag(1j * phase) + forward @ np.diag(-0.5 * rate) @ inverse
    values, vectors = np.linalg.eig(generator)
    start = np.linalg.solve(vectors, launch)
    r = np.asarray(space.r)
    aperture = r <= space.radius
    weights = np.asarray(space.radial_weights)[aperture]
    initial = inverse @ launch
    energy = float(np.sum(np.asarray(space.radial_weights) * np.abs(initial) ** 2))
    # The bare grid: the launch's own field beyond the design grid is zero.
    spacing = float(space.outer_radius) / space.n_r
    count = max(space.n_r, round(half_size / spacing))
    large = QDHT(count * spacing, count)
    outside = large.r > space.outer_radius
    series = j0(np.outer(large.r, k_r)) * (
        np.asarray(space.spectral_weights) / (2 * np.pi)
    )
    field_large = series @ launch
    field_large[outside] = 0.0
    large_diagonal = large.j1[:, None] / large.j1[None, :]
    columns = (large.forward_scale * large_diagonal * large.matrix) @ field_large
    k_large = np.asarray(large.k_r)
    large_phase = np.sqrt(k * k - k_large * k_large + 0j) - k
    evaluate = j0(np.outer(r[aperture], k_large)) * (
        np.asarray(large.spectral_weights) / (2 * np.pi)
    )
    differences = np.empty(planes.size)
    for index, z in enumerate(planes):
        collar = (inverse @ (vectors @ (np.exp(values * z) * start)))[aperture]
        bare = evaluate @ (np.exp(1j * large_phase * z) * columns)
        differences[index] = float(np.sum(weights * np.abs(collar - bare) ** 2))
    return differences, energy


def _periodic_return(  # noqa: PLR0913, PLR0917 - one frequency's comparison
    axes: tuple[np.ndarray, ...],
    sizes: tuple[float, ...],
    rate: Float[np.ndarray, " n_nodes"],
    launch: Complex[np.ndarray, " n_nodes"],
    k: float,
    planes: np.ndarray,
    step: float,
    half_sizes: tuple[float, ...],
) -> tuple[np.ndarray, float]:
    """Return one frequency's aperture difference energy at each plane, periodically.

    A Strang march on the design's own periodic grid, one or two axes, x fastest;
    the bare grid is the same spacing over `half_sizes`, exact in the wavenumber.

    Parameters
    ----------
    axes : tuple of numpy.ndarray
        The design's centred axes, m, x first.
    sizes : tuple of float
        The requested half-sizes, m.
    rate : numpy.ndarray
        The collar's power rate on the nodes, 1/m.
    launch : numpy.ndarray
        The launch's spectral columns (unshifted FFT order, x fastest).
    k : float
        The wavenumber, rad/m (``beta_p`` for a slab mode).
    planes : numpy.ndarray
        Positions, m.
    step : float
        The march's longest step, m.
    half_sizes : tuple of float
        The bare grid's half-sizes, m.

    Returns
    -------
    tuple
        The difference energy per plane, and the launch energy, in one unit.
    """
    shape = tuple(axis.size for axis in reversed(axes))
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    kappa = np.meshgrid(
        *[
            np.fft.fftfreq(axis.size) * (2 * np.pi / d)
            for axis, d in zip(reversed(axes), reversed(spacing), strict=True)
        ],
        indexing="ij",
    )
    k_perp2 = sum(part * part for part in kappa)
    phase = np.sqrt(k * k - k_perp2 + 0j) - k
    field = np.fft.ifftn(launch.reshape(shape))
    loss = rate.reshape(shape)
    grids = np.meshgrid(*reversed(axes), indexing="ij")
    aperture = np.ones(shape, dtype=bool)
    for grid, size in zip(grids, reversed(sizes), strict=True):
        aperture &= np.abs(grid) <= size
    energy = float(np.sum(np.abs(field) ** 2))
    # The bare grid, centred on the same samples.
    counts = tuple(
        max(axis.size, 2 * round(half / d))
        for axis, half, d in zip(axes, half_sizes, spacing, strict=True)
    )
    big_shape = tuple(reversed(counts))
    big = np.zeros(big_shape, dtype=np.complex128)
    offsets = tuple((b - s) // 2 for b, s in zip(big_shape, shape, strict=True))
    window = tuple(slice(o, o + s) for o, s in zip(offsets, shape, strict=True))
    big[window] = field
    big_kappa = np.meshgrid(
        *[
            np.fft.fftfreq(n) * (2 * np.pi / d)
            for n, d in zip(big_shape, reversed(spacing), strict=True)
        ],
        indexing="ij",
    )
    big_phase = np.sqrt(k * k - sum(part * part for part in big_kappa) + 0j) - k
    big_spectrum = np.fft.fftn(big)
    differences = np.empty(planes.size)
    position = 0.0
    state = field
    for index, z in enumerate(planes):
        while position < z - 1e-15 * max(z, 1.0):
            h = min(step, z - position)
            half = np.exp(0.5j * phase * h)
            state = np.fft.ifftn(half * np.fft.fftn(state))
            state = state * np.exp(-0.5 * loss * h)
            state = np.fft.ifftn(half * np.fft.fftn(state))
            position += h
        bare = np.fft.ifftn(np.exp(1j * big_phase * z) * big_spectrum)[window]
        differences[index] = float(np.sum(np.abs(state - bare)[aperture] ** 2))
    return differences, energy


def _estimate(  # noqa: PLR0913, PLR0915 - one estimate at one resolution
    space: RadialGrid | FreeGrid | PlanarGrid,
    rate: Float[np.ndarray, " n_nodes"],
    launch: Complex[np.ndarray, "n_columns n_pol n_omega"],
    dispersion: SpatialDispersion,
    z_max: float,
    width: float,
    *,
    nodes: int,
    planes: int,
    enlarge: float,
) -> tuple[float, tuple[float, ...], int]:
    """Return ``U_ret`` at one resolution of the check.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The design's grid.
    rate : numpy.ndarray
        The collar's rate on the nodes, 1/m (zeros for the bare wall).
    launch : numpy.ndarray
        The noise-free launch's spectral columns, ``(n_columns, n_pol, n_omega)``.
    dispersion : solprop.spatial.SpatialDispersion
        The medium and the temporal grid.
    z_max : float
        The run's length, m.
    width : float
        The collar's narrowest width, m.
    nodes, planes : int
        Frequency nodes and the most planes.
    enlarge : float
        The bare grid's size as a multiple of its least.

    Returns
    -------
    tuple
        ``U_ret``, the nodes in rad/s and the planes compared.
    """
    grid = dispersion.grid
    band = np.nonzero(grid.inband_mask & grid.active_mask)[0]
    weights = np.asarray(space.spectral_weights)
    power = np.sum(weights[:, None, None] * np.abs(launch[..., band]) ** 2, axis=(0, 1))
    chosen, shares = _frequency_nodes(power, nodes)
    omega = np.asarray(grid.omega[band[chosen]], dtype=np.float64)
    radial = isinstance(space, RadialGrid)
    if isinstance(space, PlanarGrid):
        kappa_columns = np.abs(np.repeat(space.k_y, space.n_p))
    elif radial:
        kappa_columns = np.asarray(space.k_r)
    else:
        kappa_columns = np.sqrt(np.asarray(space.kperp_sq))
    reach = 0.0
    for omega_j, bin_j in zip(omega, band[chosen], strict=True):
        k = (
            float(np.real(dispersion.medium.index(np.array([omega_j]))[0]))
            * omega_j
            / C
        )
        column_power = weights * np.sum(np.abs(launch[:, :, bin_j]) ** 2, axis=1)
        reach = max(reach, _reach(kappa_columns, column_power, k))
    spacing = min(z_max / planes, width / (2.0 * reach) if reach > 0.0 else math.inf)
    count = max(2, math.ceil(z_max / spacing))
    positions = np.linspace(0.0, z_max, count + 1)[1:]
    total_difference = np.zeros(positions.size)
    total_energy = 0.0
    for omega_j, bin_j, share in zip(omega, band[chosen], shares, strict=True):
        index = float(np.real(dispersion.medium.index(np.array([omega_j]))[0]))
        k = index * omega_j / C
        node_difference = np.zeros(positions.size)
        node_energy = 0.0
        for pol in range(launch.shape[1]):
            columns = launch[:, pol, bin_j]
            if not np.any(columns):
                continue
            if radial:
                assert isinstance(space, RadialGrid)
                hankel = space.hankel
                diagonal = hankel.j1[:, None] / hankel.j1[None, :]
                inverse = hankel.inverse_scale * diagonal * hankel.matrix
                physical = inverse @ columns
                extent = _extent(
                    np.asarray(space.r),
                    np.asarray(space.radial_weights) * np.abs(physical) ** 2,
                )
                half = (
                    enlarge
                    * max(
                        space.outer_radius,
                        0.5 * (space.radius + extent + reach * z_max),
                    )
                    * 2.0
                )
                half = min(half, _LARGEST * space.outer_radius)
                difference, energy = _radial_return(
                    space, rate, columns, k, positions, half
                )
            elif isinstance(space, PlanarGrid):
                # One y line per slab mode, with its own beta_p in place of k.
                difference = np.zeros(positions.size)
                energy = 0.0
                for p, mode in enumerate(space.modes):
                    beta = (
                        float(np.real(np.asarray(mode.neff(np.array([omega_j])))[0]))
                        * omega_j
                        / C
                    )
                    line = columns[p :: space.n_p]
                    y_rate = rate.reshape(space.n_y, -1)[:, 0]
                    half_y = _half(space.y, space.radius_y, line, reach, z_max, enlarge)
                    d, e = _periodic_return(
                        (np.asarray(space.y),),
                        (space.radius_y,),
                        y_rate,
                        line,
                        beta,
                        positions,
                        _step(positions, z_max),
                        (half_y,),
                    )
                    difference += d
                    energy += e
            else:
                axes = (
                    (np.asarray(space.x),)
                    if isinstance(space, LineGrid)
                    else (np.asarray(space.x), np.asarray(space.y))
                )
                sizes = (
                    (space.radius_x,)
                    if isinstance(space, LineGrid)
                    else (space.radius_x, space.radius_y)
                )
                physical = np.fft.ifftn(
                    columns.reshape(tuple(a.size for a in reversed(axes)))
                )
                halves = tuple(
                    _half(
                        axis,
                        size,
                        physical,
                        reach,
                        z_max,
                        enlarge,
                        along=len(axes) - 1 - i,
                    )
                    for i, (axis, size) in enumerate(zip(axes, sizes, strict=True))
                )
                difference, energy = _periodic_return(
                    axes,
                    sizes,
                    rate,
                    columns,
                    k,
                    positions,
                    _step(positions, z_max),
                    halves,
                )
            node_difference += difference
            node_energy += energy
        if node_energy > 0.0:
            total_difference += share * node_difference / node_energy
            total_energy += share
    returned = float(np.max(total_difference)) / max(total_energy, 1e-300)
    return returned, tuple(float(w) for w in omega), int(positions.size)


def _step(positions: np.ndarray, z_max: float) -> float:
    """Return the Strang march's longest step: the plane spacing.

    Parameters
    ----------
    positions : numpy.ndarray
        The planes, m.
    z_max : float
        The run's length, m.

    Returns
    -------
    float
        The step, m.
    """
    return z_max / positions.size


def _half(  # noqa: PLR0913 - the bare grid's half-size along one axis
    axis: np.ndarray,
    size: float,
    field: np.ndarray,
    reach: float,
    z_max: float,
    enlarge: float,
    *,
    along: int = 0,
) -> float:
    """Return the bare grid's half-size along one periodic axis (section 4.4).

    ``S >= (R + x_q + v_q L) / 2``, at least the design's own half-size, at most
    :data:`_LARGEST` times it, times `enlarge`.

    Parameters
    ----------
    axis : numpy.ndarray
        The centred axis, m.
    size : float
        The requested half-size, m.
    field : numpy.ndarray
        The launch's physical field, axes reversed (y, x).
    reach : float
        The launch's reach, ``dx/dz``.
    z_max : float
        The run's length, m.
    enlarge : float
        Multiplier of the least half-size.
    along : int, optional
        The array axis of `field` this coordinate runs along.

    Returns
    -------
    float
        The half-size, m.
    """
    profile = np.sum(
        np.abs(field) ** 2, axis=tuple(i for i in range(field.ndim) if i != along)
    )
    extent = _extent(np.abs(axis), profile)
    outer = 0.5 * float(axis.size) * float(axis[1] - axis[0])
    half = 2.0 * enlarge * max(outer, 0.5 * (size + extent + reach * z_max))
    return min(half, _LARGEST * outer)


def design_check(
    space: RadialGrid | FreeGrid | PlanarGrid,
    rate: Float[np.ndarray, " n_nodes"],
    launch: Complex[np.ndarray, "n_batch n_columns n_pol n_omega"],
    dispersion: SpatialDispersion,
    z_max: float,
    width: float,
) -> CheckResult:
    """Run the design check of SEP 40 section 4.4 on one launch.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The design's grid.
    rate : numpy.ndarray
        The collar's power rate on the nodes, 1/m.
    launch : numpy.ndarray
        The noise-free launch, canonical layout; its first batch member is checked.
    dispersion : solprop.spatial.SpatialDispersion
        The medium and the temporal grid.
    z_max : float
        The run's length, m.
    width : float
        The collar's narrowest width, m.

    Returns
    -------
    CheckResult
        The verdict, ``"pending"`` for a launch that is zero everywhere.
    """
    started = time.perf_counter()
    columns = np.asarray(launch[0], dtype=np.complex128)
    if not np.any(columns):
        return CheckResult("pending", reason="no launch to check yet")
    returned, nodes, planes = _estimate(
        space,
        rate,
        columns,
        dispersion,
        z_max,
        width,
        nodes=_NODES,
        planes=_PLANES,
        enlarge=1.0,
    )
    wall, _, _ = _estimate(
        space,
        np.zeros_like(rate),
        columns,
        dispersion,
        z_max,
        width,
        nodes=_NODES,
        planes=_PLANES,
        enlarge=1.0,
    )
    refined = False
    status: Status
    reason = ""
    if 0.5 * BUDGET <= returned <= 2.0 * BUDGET:
        refined = True
        again, nodes, planes = _estimate(
            space,
            rate,
            columns,
            dispersion,
            z_max,
            width,
            nodes=2 * _NODES,
            planes=2 * _PLANES,
            enlarge=1.5,
        )
        if abs(again - returned) > 0.25 * max(again, returned):
            status = "inconclusive"
            reason = (
                f"the refined estimate {again:.2e} disagrees with the first, "
                f"{returned:.2e}, by more than a quarter"
            )
        else:
            status = "passed" if again <= REFINED_BUDGET else "refused"
        returned = again
    else:
        status = "passed" if returned <= BUDGET else "refused"
    return CheckResult(
        status,
        returned=returned,
        bare_wall=wall,
        refined=refined,
        nodes=nodes,
        planes=planes,
        wall_time=time.perf_counter() - started,
        reason=reason,
    )

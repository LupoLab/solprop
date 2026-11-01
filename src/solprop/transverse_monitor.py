# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor's transverse edge (SEP 40 section 5.3).

A free-space run has a wall or a wrap at its grid's outer size. With no real-space
collar, whatever reaches the edge comes back, so reaching it is the failure: the
monitor judges, per face, the energy in the outer 5 % of the *outer* size, its largest
reading above its launch value, less a noise allowance, at the levels of
:class:`solprop.monitor.BoundaryCheck`. With a collar, reaching the edge is intended and
what returns is the failure; the outward shares are recorded at INFO and the collar's
removal is credited.

Reads are spaced so that light cannot cross the edge unseen between two of them,
``min(L/16, 4 g / v_q)`` with ``g`` the guard's width and ``v_q`` the reach of the
current spectrum, and the spacing is a ceiling on the step. The safety ramp's removal
is credited per read interval and judged above the realised input noise's own share.
Every run reports a ``transverse_status``; only ``clear`` is a clean verdict.

The field read is the backend's committed spectral state; the reduction is the host's,
which ``numpy_ref`` carries in phase 1 of SEP 40. A backend that does not read the edge
leaves the run ``unwatched``, which it says before the first step.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.modes.slab import PlanarGrid
from solprop.spatial_absorber import RAMP_K_FLOOR, SpatialAbsorberDesign

__all__ = ["GUARD", "READS", "TransverseEdge", "TransverseStatus"]

logger = logging.getLogger(__name__)

GUARD = 0.05
"""The edge's width as a fraction of the outer size on its axis (SEP 40 section 5.3)."""

READS = 16
"""The read spacing's ceiling is the guide's length over this many."""

NOISE_FACTOR = 8.0
"""Standard deviations of one noise reading the allowance takes (section 5.3)."""

TransverseStatus = Literal[
    "clear", "warned", "stopped", "insensitive", "coarse", "unwatched", "off"
]


@dataclass(frozen=True)
class _Face:
    """One face of the transverse edge.

    Attributes
    ----------
    name : str
        ``"r"``, ``"x-"``, ``"x+"``, ``"y-"`` or ``"y+"``.
    edge : numpy.ndarray
        Nodes (or rows) in the outer :data:`GUARD` of the outer size.
    clip : numpy.ndarray
        Nodes in the outer :data:`GUARD` of the requested size, for the launch.
    """

    name: str
    edge: np.ndarray
    clip: np.ndarray


def _faces(space: RadialGrid | FreeGrid | PlanarGrid) -> tuple[_Face, ...]:
    """Return the faces of a grid's edge on its physical nodes or rows.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The grid.

    Returns
    -------
    tuple of _Face
        One radially, four on a Cartesian box, two on a line or a slab.
    """
    if isinstance(space, RadialGrid):
        r = np.asarray(space.r)
        return (
            _Face(
                "r",
                r >= (1 - GUARD) * space.outer_radius,
                (r >= (1 - GUARD) * space.radius) & (r <= space.radius),
            ),
        )

    def axis_faces(
        name: str, axis: np.ndarray, size: float, flat: np.ndarray
    ) -> tuple[_Face, _Face]:
        """Return an axis's low and high faces.

        Parameters
        ----------
        name : str
            The axis, ``"x"`` or ``"y"``.
        axis : numpy.ndarray
            Its centred samples, m.
        size : float
            Its requested half-size, m.
        flat : numpy.ndarray
            Each node's coordinate along it, m.

        Returns
        -------
        tuple of _Face
            The two faces.
        """
        outer = 0.5 * axis.size * float(axis[1] - axis[0])
        low = _Face(
            f"{name}-",
            flat <= -(1 - GUARD) * outer,
            (flat <= -(1 - GUARD) * size) & (flat >= -size),
        )
        high = _Face(
            f"{name}+",
            flat >= (1 - GUARD) * outer - 0.5 * float(axis[1] - axis[0]),
            (flat >= (1 - GUARD) * size) & (flat <= size),
        )
        return low, high

    if isinstance(space, PlanarGrid):
        y = np.asarray(space.y)
        return axis_faces("y", y, space.radius_y, y)
    x = np.tile(np.asarray(space.x), space.n_y)
    faces = axis_faces("x", np.asarray(space.x), space.radius_x, x)
    if isinstance(space, LineGrid):
        return faces
    y = np.repeat(np.asarray(space.y), space.n_x)
    return faces + axis_faces("y", np.asarray(space.y), space.radius_y, y)


def _guard_width(space: RadialGrid | FreeGrid | PlanarGrid) -> float:
    """Return the narrowest face's guard width, m.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The grid.

    Returns
    -------
    float
        ``GUARD`` times the smallest outer half-size.
    """
    if isinstance(space, RadialGrid):
        return GUARD * space.outer_radius
    if isinstance(space, PlanarGrid):
        return GUARD * 0.5 * space.n_y * float(space.y[1] - space.y[0])
    widths = [0.5 * space.n_x * float(space.x[1] - space.x[0])]
    if not isinstance(space, LineGrid):
        widths.append(0.5 * space.n_y * float(space.y[1] - space.y[0]))
    return GUARD * min(widths)


@dataclass
class TransverseEdge:
    """The transverse edge of one free-space run: geometry, totals and record.

    Attributes
    ----------
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or PlanarGrid
        The run's grid.
    design : solprop.spatial_absorber.SpatialAbsorberDesign or None
        Its transverse boundary; ``None`` for ``spatial_absorber=False``.
    budget : float
        The warning level as a field amplitude: energy shares above its square warn.
    stop_factor : float
        Budgets above which a share stops the run.
    length : float
        The guide's length, m.
    band : numpy.ndarray
        The requested band's bins on the run's frequency axis.
    k_min : float
        The smallest wavenumber over the requested band, rad/m.
    """

    space: RadialGrid | FreeGrid | PlanarGrid
    design: SpatialAbsorberDesign | None
    budget: float
    stop_factor: float
    length: float
    band: np.ndarray
    k_min: float
    _faces: tuple[_Face, ...] = field(init=False)
    _weights: np.ndarray = field(init=False)
    _columns: np.ndarray = field(init=False)
    _kappa: np.ndarray = field(init=False)
    _guard: float = field(init=False)
    _launch: float = field(default=math.nan, init=False)
    _baseline: list[float] = field(default_factory=list, init=False)
    _peak: list[float] = field(default_factory=list, init=False)
    _peak_raw: list[float] = field(default_factory=list, init=False)
    _allowance: float = field(default=0.0, init=False)
    _noise_columns: np.ndarray | None = field(default=None, init=False)
    _noise_share_in_ramp: float = field(default=0.0, init=False)
    _ramp_raw: float = field(default=0.0, init=False)
    _ramp_noise: float = field(default=0.0, init=False)
    _spatial_removed: float = field(default=0.0, init=False)
    _last_rate: tuple[float, float] | None = field(default=None, init=False)
    _last_columns: np.ndarray | None = field(default=None, init=False)
    _last_z: float = field(default=math.nan, init=False)
    _next: float = field(default=math.inf, init=False)
    _reads: int = field(default=0, init=False)
    _largest_gap: float = field(default=0.0, init=False)
    _warned_at: float = field(default=math.nan, init=False)
    _status: TransverseStatus = field(default="clear", init=False)
    _reason: str = field(default="", init=False)
    _warned: bool = field(default=False, init=False)
    _ramp_warned: bool = field(default=False, init=False)
    _resumed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Build the faces, the node weights and the column wavenumbers."""
        space = self.space
        self._faces = _faces(space)
        self._guard = _guard_width(space)
        if isinstance(space, RadialGrid):
            self._weights = np.asarray(space.radial_weights, dtype=np.float64)
            self._kappa = np.asarray(space.k_r, dtype=np.float64)
        elif isinstance(space, PlanarGrid):
            self._weights = np.full(space.n_y, float(space.y[1] - space.y[0]))
            self._kappa = np.abs(np.repeat(np.asarray(space.k_y), space.n_p))
        else:
            self._weights = np.asarray(space.spatial_weights, dtype=np.float64)
            self._kappa = np.sqrt(np.asarray(space.kperp_sq, dtype=np.float64))
        self._columns = np.asarray(space.spectral_weights, dtype=np.float64)
        self._peak = [0.0] * len(self._faces)
        self._peak_raw = [0.0] * len(self._faces)
        self._baseline = [0.0] * len(self._faces)

    # ------------------------------------------------------------------ reductions
    def _node_power(
        self, state: Complex[np.ndarray, "n_modes n_pol n_omega"]
    ) -> Float[np.ndarray, " n_nodes"]:
        """Return each physical node's (or row's) energy, summed over pol and frequency.

        Parameters
        ----------
        state : numpy.ndarray
            One batch member's spectral columns.

        Returns
        -------
        numpy.ndarray
            ``|E|^2`` summed, unweighted, in the state's units squared.
        """
        space = self.space
        flat = state.reshape(state.shape[0], -1)
        if isinstance(space, RadialGrid):
            hankel = space.hankel
            inverse = (
                hankel.inverse_scale
                * (hankel.j1[:, None] / hankel.j1[None, :])
                * hankel.matrix
            )
            physical = inverse @ flat
            return np.sum(np.abs(physical) ** 2, axis=1)
        if isinstance(space, PlanarGrid):
            cube = flat.reshape(space.n_y, space.n_p, -1)
            physical = np.fft.ifft(cube, axis=0)
            return np.sum(np.abs(physical) ** 2, axis=(1, 2))
        cube = flat.reshape(space.n_y, space.n_x, -1)
        physical = np.fft.ifftn(cube, axes=(0, 1))
        return np.sum(np.abs(physical) ** 2, axis=2).reshape(-1)

    def _column_power(
        self, state: Complex[np.ndarray, "n_modes n_pol n_omega"]
    ) -> Float[np.ndarray, " n_columns"]:
        """Return each spectral column's energy, in the node reduction's units.

        Parameters
        ----------
        state : numpy.ndarray
            One batch member's spectral columns.

        Returns
        -------
        numpy.ndarray
            Weighted ``|E|^2``, scaled so that its sum is the node reduction's total.
        """
        return self._columns * np.sum(np.abs(state) ** 2, axis=(1, 2))

    # ------------------------------------------------------------------ the record
    def launch(
        self,
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
        *,
        noise: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
        steps: Float[np.ndarray, " n_steps"] | None,
        reads: bool,
        declared: bool,
    ) -> None:
        """Record the launch, judge it, and set the first read.

        Parameters
        ----------
        state : numpy.ndarray
            The launched state, one member, spectral columns.
        z : float
            Its position, m.
        noise : numpy.ndarray or None
            The input noise it carries, if any.
        steps : numpy.ndarray or None
            A fixed schedule's steps, which the read spacing cannot limit.
        reads : bool
            Whether the backend reads the edge.
        declared : bool
            Whether the user declared content above ``discard_above``.

        Raises
        ------
        BoundaryError
            When physical light launched into the safety ramp's band exceeds ten
            budgets (section 5.3).
        """
        nodes = self._node_power(state)
        weighted = self._weights * nodes
        if self._resumed:
            # The launch was judged by the first execution; its baselines and totals
            # are restored, and the reads continue from the resumed state.
            self._last_z = z
            if noise is not None:
                self._noise_columns = self._column_power(noise)
            self._next = z + self._spacing(state) if reads else math.inf
            self._last_columns = self._column_power(state)
            self._last_rate = (z, self._collar_rate(nodes))
            if not reads:
                self._status = "unwatched"
            return
        self._launch = float(np.sum(weighted))
        self._last_z = z
        if not self._launch > 0.0:
            self._next = math.inf
            return
        warn, stop = self.budget**2, (self.stop_factor * self.budget) ** 2
        # About the launch, not the backend: judged whether or not the edge is read.
        self._judge_ramp_band(
            state if noise is None else state - noise, stop, warn, declared=declared
        )
        if not reads:
            self._status = "unwatched"
            self._reason = "this backend does not read the transverse edge"
            warnings.warn(
                "this backend does not read the transverse edge of a free-space run: "
                "the run is unwatched there and cannot report a clean verdict; "
                "backend='numpy_ref' reads it (SEP 40 section 5.3)",
                BoundaryWarning,
                stacklevel=4,
            )
            return
        for index, face in enumerate(self._faces):
            share = float(np.sum(weighted[face.edge])) / self._launch
            self._baseline[index] = share
            clipped = float(np.sum(weighted[face.clip])) / self._launch
            if clipped > warn:
                warnings.warn(
                    f"the launch holds {clipped:.2e} of its energy in the outer "
                    f"{GUARD:.0%} of the requested size on its {face.name} face: the "
                    "aperture clips it",
                    BoundaryWarning,
                    stacklevel=4,
                )
        if noise is not None:
            noise_nodes = self._weights * self._node_power(noise)
            eta = float(np.sum(noise_nodes)) / self._launch
            share = float(np.sum(self._weights[self._faces[0].edge])) / float(
                np.sum(self._weights)
            )
            modes = max(int(np.count_nonzero(self._faces[0].edge)), 1) * max(
                int(self.band.size), 1
            )
            self._allowance = NOISE_FACTOR * eta * share / math.sqrt(modes)
            self._noise_columns = self._column_power(noise)
            if self.design is not None:
                ramp = self.design.ramp_rate > 0.0
                self._noise_share_in_ramp = (
                    float(np.sum(self._noise_columns[ramp])) / self._launch
                )
            if self._allowance > warn:
                self._status = "insensitive"
                self._reason = (
                    f"the input noise's allowance {self._allowance:.2e} exceeds the "
                    "warning"
                )
                warnings.warn(
                    "the transverse edge cannot be judged below "
                    f"{self._allowance:.2e} of the launch energy under this input "
                    "noise (SEP 40 section 5.3)",
                    BoundaryWarning,
                    stacklevel=4,
                )
        spacing = self._spacing(state)
        self._next = z + spacing
        if steps is not None and steps.size and float(np.max(steps)) > spacing:
            self._status = "coarse" if self._status == "clear" else self._status
            self._largest_gap = float(np.max(steps))
            warnings.warn(
                f"the fixed schedule's longest step, {float(np.max(steps)):.3e} m, is "
                f"longer than the transverse edge's read spacing, {spacing:.3e} m: "
                "light could cross the edge unseen between two reads; the run is "
                "reported as coarse (SEP 40 section 5.3)",
                BoundaryWarning,
                stacklevel=4,
            )
        self._last_columns = self._column_power(state)
        self._last_rate = (z, self._collar_rate(nodes))

    def _judge_ramp_band(
        self,
        launch: Complex[np.ndarray, "n_modes n_pol n_omega"],
        stop: float,
        warn: float,
        *,
        declared: bool,
    ) -> None:
        """Judge the noise-free launch's share in the safety ramp's band (5.3).

        Parameters
        ----------
        launch : numpy.ndarray
            The launch as attached, less the input noise it carries.
        stop, warn : float
            The stop and warning levels as energy shares.
        declared : bool
            Whether content above ``discard_above`` was declared.

        Raises
        ------
        BoundaryError
            Above the stop, unless declared.
        """
        if self.design is None:
            return
        columns = self._column_power(launch)
        total = float(np.sum(columns))
        share = (
            float(np.sum(columns[self.design.ramp_rate > 0.0])) / total
            if total > 0.0
            else 0.0
        )
        if declared:
            logger.info("the launch holds %.2e of its energy in the ramp's band", share)
            return
        if share > stop:
            msg = (
                f"the launch holds {share:.2e} of its energy in the safety ramp's "
                "band, which the ramp would erase: refine the transverse grid, launch "
                "with less tilt, declare the content with "
                "SpatialAbsorber(discard_above=...), or remove the ramp with "
                "spatial_absorber=False (SEP 40 section 5.3)"
            )
            raise BoundaryError(msg)
        if share > warn:
            warnings.warn(
                f"the launch holds {share:.2e} of its energy in the safety ramp's "
                "band, which the ramp removes as the run starts",
                BoundaryWarning,
                stacklevel=5,
            )

    def _collar_rate(self, nodes: Float[np.ndarray, " n_nodes"]) -> float:
        """Return a requested collar's removal rate, as a share of the launch per m.

        Parameters
        ----------
        nodes : numpy.ndarray
            Each node's unweighted energy.

        Returns
        -------
        float
            ``sum alpha w |E|^2 / U_launch``; zero without a collar.
        """
        if self.design is None or self.design.collar_rate is None:
            return 0.0
        rate = np.asarray(self.design.collar_rate)
        if isinstance(self.space, PlanarGrid):
            rate = rate.reshape(self.space.n_y, -1)[:, 0]
        return float(np.sum(rate * self._weights * nodes)) / self._launch

    def _spacing(self, state: Complex[np.ndarray, "n_modes n_pol n_omega"]) -> float:
        """Return the read spacing for the current spectrum, ``min(L/16, 4 g / v_q)``.

        Parameters
        ----------
        state : numpy.ndarray
            One member's spectral columns.

        Returns
        -------
        float
            The spacing, m.
        """
        columns = self._column_power(state)
        if self._noise_columns is not None:
            columns = np.maximum(columns - self._noise_columns, 0.0)
        order = np.argsort(self._kappa)
        tail = np.cumsum(columns[order][::-1])[::-1] / self._launch
        above = np.nonzero(tail > self.budget**2)[0]
        kappa_q = float(self._kappa[order][above[-1]]) if above.size else 0.0
        k = self.k_min
        k_z = math.sqrt(max(k * k - kappa_q * kappa_q, 0.0))
        reach = kappa_q / max(k_z, RAMP_K_FLOOR * k)
        spacing = self.length / READS
        if reach > 0.0:
            spacing = min(spacing, 4.0 * self._guard / reach)
        return spacing

    def ceiling(self, z: float) -> float:
        """Return the longest step from `z` that does not pass the next read.

        Parameters
        ----------
        z : float
            The current position, m.

        Returns
        -------
        float
            The step, m; infinite when nothing is read.
        """
        if not math.isfinite(self._next):
            return math.inf
        return max(self._next - z, 1e-15 * max(abs(z), 1.0))

    def due(self, z: float) -> bool:
        """Return whether the committed step at `z` reaches the next read.

        Parameters
        ----------
        z : float
            The committed position, m.

        Returns
        -------
        bool
            True to call :meth:`read`.
        """
        return z >= self._next - 1e-12 * max(abs(z), 1.0)

    def read(
        self, state: Complex[np.ndarray, "n_modes n_pol n_omega"], z: float
    ) -> None:
        """Read the edge at a committed position, credit the removals and judge.

        Parameters
        ----------
        state : numpy.ndarray
            One member's spectral columns.
        z : float
            Its position, m.

        Raises
        ------
        BoundaryError
            When a face's judged share exceeds ``stop_factor`` budgets.
        """
        if not self._launch > 0.0 or self._status == "unwatched":
            return
        nodes = self._node_power(state)
        weighted = self._weights * nodes
        gap = z - self._last_z
        if gap <= 0.0:
            return
        self._reads += 1
        self._largest_gap = max(self._largest_gap, gap)
        # The ramp's removal over the interval, from each column's power at its start.
        if self.design is not None and self._last_columns is not None:
            taken = -np.expm1(-self.design.ramp_rate * gap)
            self._ramp_raw += float(np.sum(self._last_columns * taken)) / self._launch
            if self._noise_columns is not None:
                self._ramp_noise += float(np.sum(self._noise_columns * taken)) / (
                    self._launch
                )
        rate = self._collar_rate(nodes)
        if self._last_rate is not None:
            previous_z, previous = self._last_rate
            self._spatial_removed += 0.5 * (z - previous_z) * (previous + rate)
        self._last_rate = (z, rate)
        self._last_z = z
        self._last_columns = self._column_power(state)
        self._next = z + self._spacing(state)
        if self.design is not None and self.design.has_collar:
            # With a collar, outward flow is intended: recorded at INFO, never stopped.
            for index, face in enumerate(self._faces):
                share = float(np.sum(weighted[face.edge])) / self._launch
                self._peak_raw[index] = max(self._peak_raw[index], share)
            return
        warn, stop = self.budget**2, (self.stop_factor * self.budget) ** 2
        for index, face in enumerate(self._faces):
            share = float(np.sum(weighted[face.edge])) / self._launch
            self._peak_raw[index] = max(self._peak_raw[index], share)
            judged = share - self._baseline[index] - self._allowance
            self._peak[index] = max(self._peak[index], judged)
            if judged > stop:
                self._status = "stopped"
                raise BoundaryError(self._stop_message(face.name, judged, z))
            if judged > warn and not self._warned:
                self._warned = True
                self._warned_at = z
                if self._status in ("clear", "coarse"):
                    self._status = "warned"
                warnings.warn(
                    f"light reached the transverse edge on its {face.name} face at "
                    f"z = {z:.4e} m: {judged:.2e} of the launch energy above its "
                    "launch value, which the wall or the wrap returns (SEP 40 "
                    "section 5.3)",
                    BoundaryWarning,
                    stacklevel=4,
                )
        self._judge_ramp_removal()

    def _judge_ramp_removal(self) -> None:
        """Warn once when the ramp has removed more than the budget (Q8 warns only)."""
        removed = self.ramp_removed
        if removed > self.budget**2 and not self._ramp_warned:
            self._ramp_warned = True
            warnings.warn(
                f"the safety ramp has removed {removed:.2e} of the launch energy above "
                "the input noise's share: content reached the grid's outer "
                "wavenumbers; refine the transverse grid (SEP 40 section 5.3)",
                BoundaryWarning,
                stacklevel=5,
            )

    def _stop_message(self, face: str, judged: float, z: float) -> str:
        """Return the stop's message, naming a size and a sample count to try.

        Parameters
        ----------
        face : str
            The face.
        judged : float
            Its judged share.
        z : float
            Where it stopped, m.

        Returns
        -------
        str
            The message.
        """
        first = self._warned_at if math.isfinite(self._warned_at) else z
        factor = self.length / max(first, 1e-300)
        space = self.space
        if isinstance(space, RadialGrid):
            size = f"a radius of at least {factor * space.outer_radius:.3g} m"
            count = f"n_r of at least {math.ceil(factor * space.n_r)}"
        elif isinstance(space, PlanarGrid):
            outer = 0.5 * space.n_y * float(space.y[1] - space.y[0])
            size = f"a y half-width of at least {factor * outer:.3g} m"
            count = f"n_y of at least {math.ceil(factor * space.n_y)}"
        else:
            outer = 0.5 * space.n_x * float(space.x[1] - space.x[0])
            size = f"an x half-width of at least {factor * outer:.3g} m"
            count = f"n_x of at least {math.ceil(factor * space.n_x)}"
        return (
            f"light reached the transverse edge on its {face} face at z = {z:.4e} m: "
            f"{judged:.2e} of the launch energy above its launch value, past the "
            f"stop of {(self.stop_factor * self.budget) ** 2:.0e}; the wall or the "
            f"wrap returns it into the aperture. Try {size}, at the same spacing "
            f"({count}); this is a lower bound, and light generated late may need "
            "more. A user who knows better raises BoundaryCheck.transverse_budget"
        )

    # ------------------------------------------------------------------ results
    @property
    def ramp_removed(self) -> float:
        """Return the safety ramp's removal above the input noise's own share.

        Returns
        -------
        float
            A share of the launch energy, signed.
        """
        return self._ramp_raw - min(self._ramp_noise, self._noise_share_in_ramp)

    @property
    def status(self) -> TransverseStatus:
        """Return the run's transverse status (section 5.3).

        Returns
        -------
        str
            ``clear``, ``warned``, ``stopped``, ``insensitive``, ``coarse`` or
            ``unwatched``.
        """
        return self._status

    def resume(self, record: dict[str, object]) -> None:
        """Continue an earlier execution's record of the edge (SEP 40 section 5.3).

        The launch baselines and the totals are restored, so a resume judges against
        the original launch; :meth:`launch` then records the resumed state unjudged.

        Parameters
        ----------
        record : dict
            The ``transverse`` entry of the earlier monitor record.
        """

        def number(name: str, default: float = 0.0) -> float:
            """Return a recorded number, or `default`.

            Parameters
            ----------
            name : str
                The field.
            default : float, optional
                What an absent field stands for.

            Returns
            -------
            float
                The value.
            """
            value = record.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                return default
            return float(value)

        def numbers(name: str) -> list[float] | None:
            """Return a recorded per-face list, or ``None`` if it does not fit.

            Parameters
            ----------
            name : str
                The field.

            Returns
            -------
            list of float or None
                One value per face.
            """
            value = record.get(name)
            if not isinstance(value, list) or len(value) != len(self._faces):
                return None
            return [float(cast("float", one)) for one in cast("list[object]", value)]

        launch = number("launch_energy", math.nan)
        baseline = numbers("edge_baseline")
        if not launch > 0.0 or baseline is None:
            return
        self._resumed = True
        self._launch = launch
        self._baseline = baseline
        self._peak = numbers("edge_max") or self._peak
        self._peak_raw = numbers("edge_max_raw") or self._peak_raw
        self._allowance = number("noise_allowance")
        self._ramp_raw = number("raw_ramp_removed")
        self._ramp_noise = number("noise_ramp_removed")
        self._noise_share_in_ramp = number("noise_share_in_ramp")
        self._spatial_removed = number("spatial_removed")
        self._reads = int(number("reads"))
        self._largest_gap = number("largest_gap")
        self._warned_at = number("warned_at", math.nan)
        self._warned = math.isfinite(self._warned_at)
        status = record.get("transverse_status")
        if isinstance(status, str) and status in (
            "clear",
            "warned",
            "insensitive",
            "coarse",
        ):
            self._status = cast("TransverseStatus", status)

    def summary(self) -> dict[str, object]:
        """Return the edge's record for the monitor's.

        Returns
        -------
        dict
            JSON-ready values: the status and its reason, the faces, the launch
            energy and each face's launch baseline, largest judged and raw shares,
            the noise allowance, the ramp's removal (judged and raw) and a collar's,
            the reads and the largest gap between them, and where it first warned.
        """
        return {
            "launch_energy": self._launch if math.isfinite(self._launch) else None,
            "noise_ramp_removed": self._ramp_noise,
            "noise_share_in_ramp": self._noise_share_in_ramp,
            "transverse_status": self._status,
            "transverse_reason": self._reason,
            "faces": [face.name for face in self._faces],
            "edge_baseline": list(self._baseline),
            "edge_max": list(self._peak),
            "edge_max_raw": list(self._peak_raw),
            "noise_allowance": self._allowance,
            "ramp_removed": self.ramp_removed,
            "raw_ramp_removed": self._ramp_raw,
            "spatial_removed": self._spatial_removed,
            "reads": self._reads,
            "largest_gap": self._largest_gap,
            "warned_at": None if math.isnan(self._warned_at) else self._warned_at,
        }

    def finish(self) -> None:
        """Log the edge's verdict at the end of a run."""
        logger.info(
            "transverse edge: %s; largest judged share %.2e; the ramp removed %.2e; "
            "%d reads",
            self._status,
            max(self._peak, default=0.0),
            self.ramp_removed,
            self._reads,
        )

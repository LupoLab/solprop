# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collect spatial host statistics through bounded transverse reconstruction chunks."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields

import numpy as np
from jaxtyping import Complex, Float

from solprop import fourier
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.profiles import ZProfile
from solprop.spatial import (
    ABSORBED_CHANNELS,
    FAR_FIELD_STATISTICS,
    OPERATOR_STATISTICS,
    ApertureTables,
    aperture_statistics,
    cartesian_fluence_widths,
    radial_fluence_widths,
)
from solprop.stats import PROFILE_STATISTICS, StatsSpec, analytic_field, profile_value
from solprop.translate.modal import Transform
from solprop.translate.problem import KernelProblem
from solprop.translate.spatial import (
    FluxPlan,
    SeparableOperator,
    SpatialGeometry,
    aperture_tables,
    axial_admittance,
    axial_factors,
)

__all__ = [
    "AXIS_FOLDED",
    "AXIS_SPECTRUM_ONLY",
    "FAR_FIELD_BAND_BUDGET",
    "LINE_BLOCK",
    "FarFieldLayout",
    "FarFieldTables",
    "collar_block",
    "collar_field_fraction",
    "collect_spatial",
    "far_field_bands",
    "far_field_layout",
    "far_field_partials",
    "radial_absorber_matrix",
    "slab_columns",
    "slab_projection",
    "slab_rows",
    "slab_synthesis",
]


def collar_field_fraction(
    problem: KernelProblem,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
) -> float:
    """Measure the fraction of the beam's field norm in the declared spatial collar.

    A named electric-field-norm boundary diagnostic, the share of
    ``integral |E_a|^2 dA dt`` on the collar's nodes: it says how much field reaches
    the transverse boundary. It is not a fraction of the physical axial flux, which
    the nonlocal admittance need not localise on those nodes (SEP 34 section 5.1).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Plain geometry, temporal scaling and scalar synthesis data.
    field : numpy.ndarray
        One internal spatial spectrum, with the problem's state dimensions.

    Returns
    -------
    float
        Field-norm fraction on nodes where the spatial window is below one. Zero
        for an empty beam, absent geometry or a disabled spatial window.

    Notes
    -----
    On the Cartesian and planar kinds the physical field is reconstructed **by block
    of frequency** (:func:`collar_block`), never as the whole volume: each node's
    fluence is a sum over frequency, so blocks accumulate it exactly, up to the order
    of the sum (SEP 22 section 6.3), and the transient is one block, bounded by the
    node chunk the run's ``node_budget`` admits. The radial kind synthesises by node
    chunk, as before.
    """
    transform = problem.transform
    geometry = transform.geometry
    if geometry is None or np.all(geometry.window == 1):
        return 0.0
    dt = problem.grids.dt_fine * problem.grids.pad_scale
    if geometry.kind in ("free", "slab"):
        fluence = _blockwise_fluence(problem, field, dt)
        energy = geometry.spatial_weights * fluence
        total = float(np.sum(energy))
        collar = float(np.sum(energy[geometry.window < 1]))
        return collar / total if total > 0 else 0.0
    total = 0.0
    collar = 0.0
    for first in range(0, transform.n_nodes, transform.node_chunk):
        rows = slice(first, first + transform.node_chunk)
        physical = (
            (transform.synthesis[rows] / problem.grids.pad_scale)
            @ field.reshape(transform.n_modes, -1)
        ).reshape(-1, field.shape[1], field.shape[2])
        # Both epsilon_0 c/2 and any common FFT factor cancel in this fraction.
        fluence = np.sum(
            fourier.energy_omega(physical, dt, kind=problem.grid_kind), axis=1
        )
        energy = geometry.spatial_weights[rows] * fluence
        total += float(np.sum(energy))
        collar += float(np.sum(energy[geometry.window[rows] < 1]))
    return collar / total if total > 0 else 0.0


def collar_block(problem: KernelProblem) -> int:
    """Return how many frequency bins the collar diagnostic reconstructs at a time.

    A block of the physical field on every node holds at most as many complex samples,
    with its working copies (the strided slice, the transform's output and the
    weighted power: three block volumes), as one node chunk of the response pass holds
    of one temporal column, ``node_chunk * n_pol * n_to``. The chunk is what the run's
    ``node_budget`` admits, so the diagnostic's transient is bounded by the same budget.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A Cartesian or planar problem.

    Returns
    -------
    int
        Bins per block, at least one and at most ``n_omega``.
    """
    transform = problem.transform
    n_pol = problem.shape[2]
    budget = transform.node_chunk * n_pol * problem.grids.n_to
    per_bin = 3 * transform.n_nodes * n_pol
    return int(min(problem.shape[3], max(1, budget // per_bin)))


def _blockwise_fluence(
    problem: KernelProblem,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    dt: float,
) -> Float[np.ndarray, " n_nodes"]:
    """Accumulate each node's fluence, up to a common factor, by block of frequency.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A Cartesian or planar problem.
    field : numpy.ndarray
        One internal spatial spectrum.
    dt : float
        Coarse sample spacing, s.

    Returns
    -------
    numpy.ndarray
        ``sum over polarisation and frequency of w(omega) |E|^2`` at each node, the
        quantity :func:`solprop.fourier.energy_omega` sums, shape ``(n_nodes,)``.
    """
    transform = problem.transform
    n_omega = field.shape[-1]
    weights = fourier.energy_weights(n_omega, dt, kind=problem.grid_kind)
    block = collar_block(problem)
    fluence = np.zeros(transform.n_nodes)
    for first in range(0, n_omega, block):
        bins = slice(first, first + block)
        physical = _free_physical(transform, field[..., bins], problem.grids.pad_scale)
        assert physical is not None
        fluence += np.sum(weights[bins] * np.abs(physical) ** 2, axis=(1, 2))
    return fluence


def radial_absorber_matrix(transform: Transform, pad_scale: float) -> np.ndarray:
    """Compose the fixed radial rate into one spectral-space matrix.

    Parameters
    ----------
    transform : solprop.translate.modal.Transform
        Scalar radial forward/inverse matrices and plain spatial rate data.
    pad_scale : float
        Temporal oversampling factor already folded into synthesis.

    Returns
    -------
    numpy.ndarray
        ``F diag(-alpha/2) B`` in m^-1, shape ``(n_r, n_r)``, or an empty
        matrix when spatial absorption is absent. No temporal scaling remains.

    Notes
    -----
    The radial rate is independent of time and z. It therefore commutes with
    temporal Fourier transforms, allowing this construction-time composition.
    It remains a term in N, separate from the material polarization prefactor.
    Cartesian rates use transverse FFTs instead of this dense matrix.
    """
    geometry = transform.geometry
    if (
        geometry is None
        or geometry.kind != "radial"
        or not np.any(geometry.absorber_rate)
    ):
        return np.empty((0, 0))
    return np.ascontiguousarray(
        transform.projection
        @ ((-0.5 * geometry.absorber_rate[:, None]) * (transform.synthesis / pad_scale))
    )


FAR_FIELD_BAND_BUDGET = 128 * 2**20
"""Bytes a compiled far-field pass's band partials may hold (SEP 23 section 6).

The band count is ``min(n_units, budget // (8 n_slots n_omega))``, at least one: fixed
by the shapes and this constant, never by the thread count, so the order of every sum
is the same whatever the pool.
"""

LINE_BLOCK = 64
"""Columns per unit of a line's far-field pass: a plane's unit is a whole row."""

AXIS_FOLDED = frozenset({"on_axis_spectrum", "on_axis_intensity"})
"""The axis statistics ``numba_cpu``'s far-field pass forms from its ``W = 1`` slot."""

AXIS_SPECTRUM_ONLY = frozenset({"on_axis_spectrum"})
"""The axis statistics Kokkos's far-field pass forms from its ``W = 1`` slot.

Kokkos forms ``on_axis_intensity`` in its own spatial statistics pass, so its slot
count leaves that name out.
"""


@dataclass(frozen=True)
class FarFieldLayout:
    """Where each quantity of a compiled far-field pass lives in its slots.

    The slot order of SEP 23 section 6, present slots only: the beam spectrum, the
    axial flux, the four absorbed-power channels, the regions, then per aperture its
    collected spectrum and the real and imaginary parts of its on-axis amplitude per
    polarisation, then the on-axis amplitude itself (the ``W = 1`` slot). An absent
    quantity is ``-1``.

    Attributes
    ----------
    beam, poynting, absorbed, regions : int
        The first slot of each, or -1.
    apertures : int
        The first aperture's first slot, or -1.
    stride : int
        Slots per aperture.
    collected, amplitude : int
        Offsets within an aperture's slots, or -1.
    axis : int
        The first of the ``2 n_pol`` slots of the on-axis amplitude, or -1.
    n_slots : int
        Slots in all, positive.
    """

    beam: int
    poynting: int
    absorbed: int
    regions: int
    apertures: int
    stride: int
    collected: int
    amplitude: int
    axis: int
    n_slots: int

    def array(self) -> np.ndarray:
        """Return the layout as the integer array a compiled pass reads.

        Returns
        -------
        numpy.ndarray
            The attributes in their declared order, int64, shape ``(10,)``.
        """
        return np.array([getattr(self, item.name) for item in fields(self)], np.int64)


def far_field_layout(
    names: Collection[str],
    *,
    n_apertures: int,
    n_regions: int,
    n_pol: int,
    axis_folded: Collection[str] = AXIS_FOLDED,
) -> FarFieldLayout | None:
    """Lay out the slots a compiled far-field pass accumulates for a request.

    Parameters
    ----------
    names : collections.abc.Collection of str
        The statistics asked for; any that the pass does not form are ignored.
    n_apertures, n_regions : int
        The request's table rows.
    n_pol : int
        The state's polarisation components.
    axis_folded : collections.abc.Collection of str, optional
        The axis statistics that read the ``W = 1`` slot: :data:`AXIS_FOLDED` on
        ``numba_cpu`` (the default), :data:`AXIS_SPECTRUM_ONLY` on the native backend.

    Returns
    -------
    FarFieldLayout or None
        The layout, or ``None`` when the pass has nothing to form.

    Examples
    --------
    >>> from solprop.backends.spatial import far_field_layout
    >>> layout = far_field_layout(
    ...     ("beam_spectrum", "aperture_spectrum", "on_axis_spectrum"),
    ...     n_apertures=2, n_regions=0, n_pol=1,
    ... )
    >>> layout.beam, layout.apertures, layout.stride, layout.axis, layout.n_slots
    (0, 1, 1, 3, 5)
    """
    wanted = set(names)
    collected = 0 if "aperture_spectrum" in wanted else -1
    amplitude = (
        (1 if collected == 0 else 0) if "aperture_on_axis_spectrum" in wanted else -1
    )
    stride = int(collected == 0) + (2 * n_pol if amplitude >= 0 else 0)
    counts = (
        ("beam", 1 if "beam_spectrum" in wanted else 0),
        ("poynting", 1 if "poynting_spectrum" in wanted else 0),
        (
            "absorbed",
            len(ABSORBED_CHANNELS) if "absorbed_power" in wanted else 0,
        ),
        ("regions", n_regions if "region_spectrum" in wanted else 0),
        ("apertures", stride * n_apertures),
        ("axis", 2 * n_pol if wanted & set(axis_folded) else 0),
    )
    first: dict[str, int] = {}
    slots = 0
    for quantity, count in counts:
        first[quantity] = slots if count else -1
        slots += count
    if slots == 0:
        return None
    return FarFieldLayout(
        **first,
        stride=stride,
        collected=collected,
        amplitude=amplitude,
        n_slots=slots,
    )


def far_field_bands(n_x: int, n_y: int, n_slots: int, n_omega: int) -> np.ndarray:
    """Return the column edges of a compiled far-field pass's bands.

    The units are whole rows on a plane and blocks of :data:`LINE_BLOCK` columns on a
    line (``n_y == 1``); the bands are contiguous runs of units, as many as the units
    or as :data:`FAR_FIELD_BAND_BUDGET` admits, whichever is fewer, and at least one.
    Nothing here depends on the thread count.

    Parameters
    ----------
    n_x, n_y : int
        The transverse grid's samples; column ``ix + n_x*iy``.
    n_slots : int
        The pass's slots.
    n_omega : int
        Frequency bins.

    Returns
    -------
    numpy.ndarray
        Ascending column edges, int64, shape ``(n_bands + 1,)``, from 0 to
        ``n_x*n_y``.

    Examples
    --------
    >>> from solprop.backends.spatial import far_field_bands
    >>> far_field_bands(4, 3, 2, 8).tolist()
    [0, 4, 8, 12]
    >>> far_field_bands(130, 1, 2, 8).tolist()
    [0, 64, 128, 130]
    """
    n_columns = n_x * n_y
    unit = n_x if n_y > 1 else LINE_BLOCK
    n_units = -(-n_columns // unit)
    n_bands = max(1, min(n_units, FAR_FIELD_BAND_BUDGET // (8 * n_slots * n_omega)))
    units = (np.arange(n_bands + 1, dtype=np.int64) * n_units) // n_bands
    return np.minimum(units * unit, n_columns)


def far_field_partials(
    problem: KernelProblem, *, axis_folded: Collection[str] = AXIS_FOLDED
) -> tuple[int, int]:
    """Return the slots and bands a compiled far-field pass holds for a problem.

    Sized from the problem's own request at construction: its names, tables and
    polarisations. A Cartesian or line problem whose request forms nothing in the pass
    (and any other kind) holds none.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The translated problem.
    axis_folded : collections.abc.Collection of str, optional
        The axis statistics the backend's pass forms, as for :func:`far_field_layout`.

    Returns
    -------
    tuple of int
        ``(n_slots, n_bands)``, both zero when nothing is held. The partials hold
        ``n_bands * n_slots * n_omega`` float64 values and their band sum
        ``n_slots * n_omega``.
    """
    geometry = problem.transform.geometry
    if geometry is None or geometry.kind != "free":
        return 0, 0
    spec = problem.stats_spec
    layout = far_field_layout(
        spec.names,
        n_apertures=spec.n_apertures,
        n_regions=spec.n_regions,
        n_pol=problem.shape[2],
        axis_folded=axis_folded,
    )
    if layout is None:
        return 0, 0
    n_x, n_y = geometry.shape
    edges = far_field_bands(n_x, n_y, layout.n_slots, problem.shape[3])
    return layout.n_slots, edges.size - 1


class FarFieldTables:
    """Hold a problem's far-field tables, formed at construction, for its statistics.

    A backend builds one when it is constructed, which forms
    :func:`solprop.translate.spatial.aperture_tables` for the problem's own spec --
    refusing a geometry the far-field statistics do not admit before any step -- and
    hands :meth:`for_spec` the spec of each statistics call. The periodic channel and
    the reference backend's filtered specs carry the problem's own apertures and
    regions, and get the held tables; a different spec handed to the run loop gets
    tables formed for it.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem the backend executes.

    Raises
    ------
    OutputError
        If the problem's spec names a far-field statistic its geometry does not
        admit (:func:`solprop.translate.spatial.aperture_tables`).
    """

    def __init__(self, problem: KernelProblem) -> None:
        """Form the problem's own tables once."""
        self._problem = problem
        self._spec = problem.stats_spec
        self._held = (
            None
            if problem.transform.geometry is None
            else aperture_tables(self._spec, problem.grids, problem.transform.geometry)
        )

    def for_spec(self, spec: StatsSpec) -> ApertureTables | None:
        """Return the tables a statistics call's spec reads.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            The spec of this call.

        Returns
        -------
        solprop.spatial.ApertureTables or None
            The held tables when the spec carries the problem's own apertures and
            regions and they were formed, tables formed for this spec otherwise, and
            ``None`` when it names no far-field statistic.
        """
        if not set(spec.names) & set(FAR_FIELD_STATISTICS):
            return None
        if (
            self._held is not None
            and spec.apertures is self._spec.apertures
            and np.array_equal(spec.spectral_regions, self._held.regions)
        ):
            return self._held
        return aperture_tables(
            spec, self._problem.grids, self._problem.transform.geometry
        )


def collect_spatial(  # noqa: PLR0913 - statistics context plus borrowed FFT workspace
    spec: StatsSpec,
    grid: TimeGrid,
    transform: Transform,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    z: float,
    flux: FluxPlan | None,
    profiles: Mapping[str, ZProfile] | None = None,
    physical: np.ndarray | None = None,
    tables: ApertureTables | None = None,
    operator: SeparableOperator | None = None,
    axis: Complex[np.ndarray, "n_pol n_omega"] | None = None,
) -> dict[str, np.ndarray]:
    """Collect spatial statistics without retaining a full physical-space field.

    Every field statistic is the axial Poynting flux of the medium at the plane
    (SEP 34): energy and spectra weigh each column and bin by the real admittance
    ``c Re(k_z)/omega`` of the plane's root, and local statistics pair the field
    with its magnetic closure, synthesised from the complex admittance.

    Parameters
    ----------
    spec : solprop.stats.StatsSpec
        Requested spatial saved-field statistics or scalar profiles.
    grid : solprop.grid.TimeGrid
        Temporal reconstruction and spectral energy measure.
    transform : solprop.translate.modal.Transform
        Scalar radial synthesis and plain geometry data from the seam.
    field : numpy.ndarray
        One internal spatial spectrum, shape ``(n_modes, n_pol, n_omega)``.
    z : float
        Current propagation position in m.
    flux : solprop.translate.spatial.FluxPlan or None
        The run's plane measure; required by every field statistic.
    profiles : collections.abc.Mapping or None, optional
        Named scalar paths, where requested.
    physical : numpy.ndarray or None, optional
        Borrowed Cartesian physical-space spectrum, reconstructed into a compiled
        backend's persistent scratch. Shape agrees with ``field``.
    tables : solprop.spatial.ApertureTables or None, optional
        The far-field tables (:class:`FarFieldTables`), needed by the far-field
        statistics of SEP 23.
    operator : solprop.translate.spatial.SeparableOperator or None, optional
        The run's separable linear operator, whose rates ``absorbed_power`` needs.
    axis : numpy.ndarray or None, optional
        The on-axis amplitude ``sum_c alpha_c E_c`` in V/m, shape
        ``(n_pol, n_omega)``, when a compiled pass has formed it (SEP 23 section 6);
        the axis statistics then read it instead of forming it here.

    Returns
    -------
    dict
        Scalar or small array values, without the backend's batch axis.

    Raises
    ------
    BackendError
        If geometry, the plane measure or a requested statistic dependency is
        missing, or a width is asked of a fluence negative beyond its rounding.

    Notes
    -----
    The far-field statistics are one pass of
    :func:`solprop.spatial.aperture_statistics` over the columns, in chunks of the
    transform's ``node_chunk``, reading the plane's root through
    :func:`solprop.translate.spatial.axial_factors`, the root the stages use.
    """
    geometry = transform.geometry
    if geometry is None:
        msg = "spatial statistics require spatial geometry"
        raise BackendError(msg)
    values: dict[str, np.ndarray] = {}
    wanted = set(spec.names)
    fielded = wanted - set(PROFILE_STATISTICS)
    if fielded and flux is None:
        msg = f"spatial statistics {sorted(fielded)} need the run's plane measure"
        raise BackendError(msg)
    plane = None if flux is None else flux.at(z)
    conversion = 0.5 * EPS0 * C
    chunks = [
        slice(first, min(first + transform.node_chunk, transform.n_modes))
        for first in range(0, transform.n_modes, transform.node_chunk)
    ]
    closure = None
    if plane is not None and wanted & {
        "on_axis_spectrum",
        "on_axis_intensity",
        "peak_intensity",
        "beam_radius",
        "fwhm_r",
    }:
        # The internal spectrum carries the conjugate admittance (SpatialFlux).
        closure = np.empty_like(field)
        for columns in chunks:
            closure[columns] = (
                np.conj(axial_admittance(plane, grid.omega, columns))[:, None]
                * field[columns]
            )
    if "energy" in wanted:
        assert plane is not None
        weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
        total = 0.0
        for columns in chunks:
            real = np.real(axial_admittance(plane, grid.omega, columns))
            density = np.sum(np.abs(field[columns]) ** 2, axis=1)
            total += float(
                np.sum(
                    geometry.spectral_weights[columns, None] * weights * real * density
                )
            )
        values["energy"] = np.asarray(conversion * total)
    values.update(
        _far_field(
            wanted, grid, transform, field, tables, plane=plane, operator=operator
        )
    )
    axis_needed = bool(
        wanted & {"on_axis_spectrum", "on_axis_intensity", "beam_radius", "fwhm_r"}
    )
    axis_fluence = 0.0
    if axis_needed:
        assert closure is not None
        axis_values, axis_fluence = _axis_statistics(
            wanted,
            grid,
            np.einsum("r,rpw->pw", geometry.on_axis, field) if axis is None else axis,
            np.einsum("r,rpw->pw", geometry.on_axis, closure),
        )
        values.update(axis_values)
    if wanted & {"peak_intensity", "beam_radius", "fwhm_r"}:
        assert closure is not None
        values.update(
            _local_statistics(
                wanted,
                grid,
                transform,
                field,
                closure,
                physical=physical,
                axis_fluence=axis_fluence,
            )
        )
    for name in wanted & set(PROFILE_STATISTICS):
        if profiles is None or name not in profiles:
            msg = f"spatial statistic {name!r} needs its named profile"
            raise BackendError(msg)
        values[name] = profile_value(profiles[name], z)
    missing = wanted - values.keys()
    if missing:
        msg = f"statistics {sorted(missing)} are not spatial saved-field statistics"
        raise BackendError(msg)
    return values


def _local_statistics(  # noqa: PLR0913 - the local pass's whole context
    wanted: set[str],
    grid: TimeGrid,
    transform: Transform,
    field: np.ndarray,
    closure: np.ndarray,
    *,
    physical: np.ndarray | None,
    axis_fluence: float,
) -> dict[str, np.ndarray]:
    """Form the peak local axial flux and the fluence widths, node chunk by chunk.

    Parameters
    ----------
    wanted : set of str
        Requested catalogue names.
    grid : solprop.grid.TimeGrid
        Temporal transform and energy measure.
    transform : solprop.translate.modal.Transform
        The run's synthesis and geometry.
    field : numpy.ndarray
        One internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
    closure : numpy.ndarray
        Its magnetic closure ``G / (eps0 c)``, same shape.
    physical : numpy.ndarray or None
        A compiled backend's borrowed physical-space field, or None.
    axis_fluence : float
        The on-axis fluence in J/m^2, for the radial widths.

    Returns
    -------
    dict
        ``peak_intensity``, ``beam_radius`` and ``fwhm_r`` where requested.

    Raises
    ------
    BackendError
        If a width is asked of a fluence negative beyond its rounding budget.
    """
    geometry = transform.geometry
    assert geometry is not None
    conversion = 0.5 * EPS0 * C
    full = (
        _free_physical(transform, field, grid.pad_scale)
        if physical is None
        else physical
    )
    full_closure = _free_physical(transform, closure, grid.pad_scale)
    fluence = np.empty(transform.n_nodes)
    budget = np.empty(transform.n_nodes)
    peak = -np.inf
    for first in range(0, transform.n_nodes, transform.node_chunk):
        rows = slice(first, first + transform.node_chunk)

        electric = analytic_field(grid, _rows(transform, grid, field, full, rows))
        magnetic = analytic_field(
            grid, _rows(transform, grid, closure, full_closure, rows)
        )
        intensity = conversion * np.sum(np.real(electric * np.conj(magnetic)), axis=1)
        fluence[rows] = grid.dt * np.sum(intensity, axis=-1)
        magnitude = (
            conversion
            * grid.dt
            * np.sum(np.abs(electric) * np.abs(magnetic), axis=(1, 2))
        )
        # A node's fluence sums n_pol * n_t products of two syntheses over the
        # columns; a few rounding units per term bound it relative to its magnitude.
        budget[rows] = (
            8
            * (field.shape[1] * grid.n_t + transform.n_modes)
            * np.finfo(float).eps
            * magnitude
        )
        peak = max(peak, float(np.max(intensity)))
    values: dict[str, np.ndarray] = {}
    if "peak_intensity" in wanted:
        values["peak_intensity"] = np.asarray(peak)
    if wanted & {"beam_radius", "fwhm_r"}:
        if np.any(fluence < -budget):
            worst = int(np.argmin(fluence + budget))
            msg = (
                "beam_radius and fwhm_r need a nonnegative fluence profile, and the "
                f"signed axial flux is {fluence[worst]:.3e} J/m^2 at node {worst}, "
                "beyond its rounding budget; the width is undefined for it"
            )
            raise BackendError(msg)
        fluence = np.maximum(fluence, 0.0)
        widths = (
            radial_fluence_widths(
                geometry.axes[0], fluence, geometry.spatial_weights, axis_fluence
            )
            if geometry.kind == "radial"
            else _slab_widths(geometry, fluence)
            if geometry.kind == "slab"
            else cartesian_fluence_widths(geometry.axes[0], geometry.axes[1], fluence)
        )
        if "beam_radius" in wanted:
            values["beam_radius"] = np.asarray(widths[:2])
        if "fwhm_r" in wanted:
            values["fwhm_r"] = np.asarray(widths[2])
    return values


def _rows(
    transform: Transform,
    grid: TimeGrid,
    source: np.ndarray,
    whole: np.ndarray | None,
    rows: slice,
) -> np.ndarray:
    """Return one node chunk of a spectrum's physical field.

    Parameters
    ----------
    transform : solprop.translate.modal.Transform
        The run's synthesis.
    grid : solprop.grid.TimeGrid
        The temporal grid, whose pad scale the synthesis carries.
    source : numpy.ndarray
        Internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
    whole : numpy.ndarray or None
        Its already synthesised physical volume, or None to synthesise the rows.
    rows : slice
        The node chunk.

    Returns
    -------
    numpy.ndarray
        Physical spectra in V/m at the chunk's nodes.
    """
    if whole is not None:
        return whole[rows]
    return (
        (transform.synthesis[rows] / grid.pad_scale)
        @ source.reshape(transform.n_modes, -1)
    ).reshape(-1, source.shape[1], source.shape[2])


def _far_field(  # noqa: PLR0913 - the far-field pass's whole context
    names: set[str],
    grid: TimeGrid,
    transform: Transform,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    tables: ApertureTables | None,
    *,
    plane: SeparableOperator | None,
    operator: SeparableOperator | None,
) -> dict[str, np.ndarray]:
    """Form the requested far-field statistics in one chunked pass.

    Parameters
    ----------
    names : set of str
        The statistics asked for; those that are not far-field statistics are left
        to the other passes.
    grid : solprop.grid.TimeGrid
        The temporal grid.
    transform : solprop.translate.modal.Transform
        The Cartesian transform, whose ``node_chunk`` bounds the pass's chunks.
    field : numpy.ndarray
        One internal spectrum, shape ``(n_x*n_y, n_pol, n_omega)``.
    tables : solprop.spatial.ApertureTables or None
        The far-field tables.
    plane : solprop.translate.spatial.SeparableOperator or None
        The plane's coefficients, whose root weighs every statistic.
    operator : solprop.translate.spatial.SeparableOperator or None
        The run's separable operator, whose rates ``absorbed_power`` reads.

    Returns
    -------
    dict
        The requested far-field statistics, empty when none is asked for.

    Raises
    ------
    BackendError
        If the tables, or the operator an operator statistic needs, are missing.
    """
    names = names & set(FAR_FIELD_STATISTICS)
    if not names:
        return {}
    if tables is None:
        msg = (
            f"the far-field statistics {sorted(names)} need the tables the backend "
            "forms at construction"
        )
        raise BackendError(msg)
    if names & set(OPERATOR_STATISTICS) and operator is None:
        msg = (
            f"{sorted(names & set(OPERATOR_STATISTICS))} need the run's separable "
            "linear operator, and this problem carries none"
        )
        raise BackendError(msg)
    assert plane is not None
    rated = operator if operator is not None else plane

    def axial(columns: slice) -> tuple[np.ndarray, np.ndarray]:
        """Return the plane's admittance and the operator's rates for a chunk."""
        _, rates = axial_factors(rated, grid.omega, columns)
        return axial_admittance(plane, grid.omega, columns), rates

    return aperture_statistics(
        sorted(names), grid, field, tables, axial=axial, chunk=transform.node_chunk
    )


def _axis_statistics(
    wanted: set[str],
    grid: TimeGrid,
    axis: np.ndarray,
    closure: np.ndarray,
) -> tuple[dict[str, np.ndarray], float]:
    """Reduce the requested statistics of the on-axis spectrum.

    Parameters
    ----------
    wanted : set of str
        Requested catalogue names.
    grid : solprop.grid.TimeGrid
        Temporal transform and energy measure.
    axis : numpy.ndarray
        The on-axis amplitude, the on-axis weights contracted with the spectrum,
        shape ``(n_pol, n_omega)``.
    closure : numpy.ndarray
        The on-axis magnetic closure ``G / (eps0 c)``, same shape.

    Returns
    -------
    tuple of dict and float
        Requested on-axis statistics and axial fluence in J/m^2.
    """
    conversion = 0.5 * EPS0 * C
    factor = fourier.spectral_density(
        np.ones(grid.n_omega, complex), grid.dt, kind=grid.kind
    )
    density = conversion * factor * np.sum(np.real(axis * np.conj(closure)), axis=0)
    values: dict[str, np.ndarray] = {}
    if "on_axis_spectrum" in wanted:
        values["on_axis_spectrum"] = density
    if "on_axis_intensity" in wanted:
        power = np.sum(
            np.real(
                analytic_field(grid, axis) * np.conj(analytic_field(grid, closure))
            ),
            axis=0,
        )
        values["on_axis_intensity"] = np.asarray(conversion * np.max(power))
    return values, float(np.sum(density) * grid.d_omega)


def _free_physical(
    transform: Transform, field: np.ndarray, pad_scale: float
) -> np.ndarray | None:
    """Reconstruct one coarse spectral volume for Cartesian or planar statistics.

    Parameters
    ----------
    transform : solprop.translate.modal.Transform
        Plain Cartesian axes and FFT shape, the planar slab matrices and y axis, or
        another transform kind.
    field : numpy.ndarray
        Internal spectrum, with flattened transverse columns first.
    pad_scale : float
        The temporal pad scale a slab synthesis carries, divided out here.

    Returns
    -------
    numpy.ndarray or None
        Physical coarse spectra in V/m at the nodes, or None for a dense transform.
    """
    geometry = transform.geometry
    if geometry is None or geometry.kind not in ("free", "slab"):
        return None
    if geometry.kind == "slab":
        return slab_synthesis(transform, field) / pad_scale
    shape = (geometry.shape[1], geometry.shape[0], *field.shape[1:])
    return np.fft.ifft2(field.reshape(shape), axes=(0, 1)).reshape(field.shape)


def slab_rows(
    transform: Transform, columns: Complex[np.ndarray, "n_modes ..."]
) -> Complex[np.ndarray, "n_modes ..."]:
    """Take a planar state from its ``(p, k_y)`` columns to its ``(p, y)`` rows.

    The inverse raw DFT in y alone, scaled by ``1/n_y``. A row is a modal amplitude
    in sqrt(W/m) at one y sample, whose time signal is real on a field grid, so every
    per-row temporal operation -- a temporal absorber, the y collar -- is exact here.

    Parameters
    ----------
    transform : solprop.translate.modal.Transform
        A transform with ``"slab"`` geometry.
    columns : numpy.ndarray
        Spectra with the ``n_p * n_y`` columns first, the channel fastest.

    Returns
    -------
    numpy.ndarray
        The same shape, rows ``p + n_p*iy``.
    """
    geometry = transform.geometry
    assert geometry is not None
    n_p = transform.synthesis.shape[1]
    shape = (geometry.shape[1], n_p, *columns.shape[1:])
    return np.fft.ifft(columns.reshape(shape), axis=0).reshape(columns.shape)


def slab_columns(
    transform: Transform, rows: Complex[np.ndarray, "n_modes ..."]
) -> Complex[np.ndarray, "n_modes ..."]:
    """Take a planar state from its ``(p, y)`` rows back to ``(p, k_y)`` columns.

    Parameters
    ----------
    transform : solprop.translate.modal.Transform
        A transform with ``"slab"`` geometry.
    rows : numpy.ndarray
        Spectra with the ``n_p * n_y`` rows first, the channel fastest.

    Returns
    -------
    numpy.ndarray
        The unscaled forward raw DFT in y, the same shape.
    """
    geometry = transform.geometry
    assert geometry is not None
    n_p = transform.synthesis.shape[1]
    shape = (geometry.shape[1], n_p, *rows.shape[1:])
    return np.fft.fft(rows.reshape(shape), axis=0).reshape(rows.shape)


def slab_synthesis(
    transform: Transform, columns: Complex[np.ndarray, "n_modes ..."]
) -> Complex[np.ndarray, "n_nodes ..."]:
    """Synthesise a planar state's field at the ``(x, y)`` nodes.

    The inverse DFT in y, then the slab synthesis in x -- the two commute, and this
    order transforms the smaller axis first. The synthesis carries the temporal pad
    scale, so the result is the field the response pass needs on the fine grid.

    Parameters
    ----------
    transform : solprop.translate.modal.Transform
        A transform with ``"slab"`` geometry.
    columns : numpy.ndarray
        Spectra with the ``n_p * n_y`` columns first.

    Returns
    -------
    numpy.ndarray
        Spectra at the nodes ``ix + n_x*iy``, shape ``(n_x*n_y, ...)``.
    """
    geometry = transform.geometry
    assert geometry is not None
    n_x, n_y = geometry.shape
    rows = slab_rows(transform, columns).reshape(n_y, transform.synthesis.shape[1], -1)
    # One real GEMM per y row, broadcast over the rows: (n_x, n_p) @ (n_p, R).
    physical = np.matmul(transform.synthesis, rows)
    return physical.reshape(n_x * n_y, *columns.shape[1:])


def slab_projection(
    transform: Transform, physical: Complex[np.ndarray, "n_nodes ..."]
) -> Complex[np.ndarray, "n_modes ..."]:
    """Project spectra at the planar nodes onto the ``(p, k_y)`` columns.

    Parameters
    ----------
    transform : solprop.translate.modal.Transform
        A transform with ``"slab"`` geometry.
    physical : numpy.ndarray
        Spectra at the nodes ``ix + n_x*iy``, first axis ``n_x*n_y``.

    Returns
    -------
    numpy.ndarray
        The slab projection in x, then the forward DFT in y, first axis
        ``n_p*n_y``.
    """
    geometry = transform.geometry
    assert geometry is not None
    n_x, n_y = geometry.shape
    n_p = transform.synthesis.shape[1]
    rows = np.matmul(transform.projection, physical.reshape(n_y, n_x, -1)).reshape(
        n_p * n_y, *physical.shape[1:]
    )
    return slab_columns(transform, rows)


def _slab_widths(geometry: SpatialGeometry, fluence: np.ndarray) -> tuple[float, ...]:
    """Measure a planar beam's widths in y from its fluence integrated over x.

    Parameters
    ----------
    geometry : solprop.translate.spatial.SpatialGeometry
        ``"slab"`` geometry, whose node weights carry the x rule.
    fluence : numpy.ndarray
        Fluence in J/m^2 at the nodes ``ix + n_x*iy``.

    Returns
    -------
    tuple of float
        The line widths in y (:func:`solprop.spatial.cartesian_fluence_widths` on
        one row): the moment half-width, the 1/e^2 half-width and the FWHM, in m,
        of the fluence integrated across the gap.
    """
    n_x, n_y = geometry.shape
    across = np.sum((geometry.spatial_weights * fluence).reshape(n_y, n_x), axis=1)
    return cartesian_fluence_widths(geometry.axes[1], np.zeros(1), across)

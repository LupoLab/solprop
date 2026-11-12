# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collect spatial host statistics through bounded transverse reconstruction chunks."""

import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, fields

import numpy as np
from jaxtyping import Complex, Float

from solprop import fourier
from solprop.backends.node_chunk import plan_node_chunk
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError, GridError
from solprop.grid import TimeGrid
from solprop.profiles import ZProfile
from solprop.spatial import (
    ABSORBED_CHANNELS,
    FAR_FIELD_STATISTICS,
    OPERATOR_STATISTICS,
    ApertureTables,
    _fluence_rings,
    _ring_fluence_widths,
    aperture_statistics,
    cartesian_fluence_widths,
    radial_fluence_widths,
)
from solprop.stats import PROFILE_STATISTICS, StatsSpec, analytic_field, profile_value
from solprop.translate.modal import Transform
from solprop.translate.problem import PropagationProblem
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
    "fluence_budget",
    "plane_fluence_widths",
    "radial_absorber_matrix",
    "signed_fluence_widths",
    "slab_columns",
    "slab_projection",
    "slab_rows",
    "slab_synthesis",
]


def collar_field_fraction(
    problem: PropagationProblem,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    node_chunk: int | None = None,
) -> float:
    """Measure the fraction of the beam's field norm in the declared spatial collar.

    A named electric-field-norm boundary diagnostic, the share of
    ``integral |E_a|^2 dA dt`` on the collar's nodes: it says how much field reaches
    the transverse boundary. It is not a fraction of the physical axial flux, which
    the nonlocal admittance need not localise on those nodes (SEP 34 section 5.1).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Plain geometry, temporal scaling and scalar synthesis data.
    field : numpy.ndarray
        One internal spatial spectrum, with the problem's state dimensions.
    node_chunk : int or None, optional
        The node chunk that bounds the transient; ``None`` takes the host plan's
        automatic chunk (:func:`solprop.backends.node_chunk.plan_node_chunk`).

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
    node chunk. The radial kind synthesises by node chunk, as before.
    """
    transform = problem.transform
    chunk = plan_node_chunk(problem).chunk if node_chunk is None else node_chunk
    geometry = transform.geometry
    if geometry is None or np.all(geometry.window == 1):
        return 0.0
    dt = problem.grids.dt_fine * problem.grids.pad_scale
    if geometry.kind in ("free", "slab"):
        fluence = _blockwise_fluence(problem, field, dt, chunk)
        energy = geometry.spatial_weights * fluence
        total = float(np.sum(energy))
        collar = float(np.sum(energy[geometry.window < 1]))
        return collar / total if total > 0 else 0.0
    total = 0.0
    collar = 0.0
    for first in range(0, transform.n_nodes, chunk):
        rows = slice(first, first + chunk)
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


def collar_block(problem: PropagationProblem, node_chunk: int | None = None) -> int:
    """Return how many frequency bins the collar diagnostic reconstructs at a time.

    A block of the physical field on every node holds at most as many complex samples,
    with its working copies (the strided slice, the transform's output and the
    weighted power: three block volumes), as one node chunk of the response pass holds
    of one temporal column, ``node_chunk * n_pol * n_to``. The chunk is what the run's
    node budget admits, so the diagnostic's transient is bounded by the same budget.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        A Cartesian or planar problem.
    node_chunk : int or None, optional
        The backend's node chunk; ``None`` takes the host plan's automatic chunk
        (:func:`solprop.backends.node_chunk.plan_node_chunk`).

    Returns
    -------
    int
        Bins per block, at least one and at most ``n_omega``.
    """
    transform = problem.transform
    n_pol = problem.shape[1]
    chunk = plan_node_chunk(problem).chunk if node_chunk is None else node_chunk
    budget = chunk * n_pol * problem.grids.n_to
    per_bin = 3 * transform.n_nodes * n_pol
    return int(min(problem.shape[2], max(1, budget // per_bin)))


def _blockwise_fluence(
    problem: PropagationProblem,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    dt: float,
    node_chunk: int,
) -> Float[np.ndarray, " n_nodes"]:
    """Accumulate each node's fluence, up to a common factor, by block of frequency.

    - ``field``: one internal spatial spectrum.
    - ``dt``: coarse sample spacing, s.
    """
    transform = problem.transform
    n_omega = field.shape[-1]
    weights = fourier.energy_weights(n_omega, dt, kind=problem.grid_kind)
    block = collar_block(problem, node_chunk)
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
"""The axis statistics a far-field pass forms from a ``W = 1`` slot."""


@dataclass(frozen=True)
class FarFieldLayout:
    """Where each quantity of a compiled far-field pass lives in its slots.

    The slot order of SEP 23 section 6, present slots only: the beam spectrum, the
    five absorbed-power channels, the regions, then per aperture its collected
    spectrum and its on-axis amplitude, then the on-axis amplitude itself (the
    ``W = 1`` slot). An amplitude is the real and imaginary parts per polarisation of
    the electric sum ``sum_c alpha_c W E_c``, then those of its magnetic closure
    ``sum_c alpha_c W conj(eta_c) E_c`` (SEP 34), ``4 n_pol`` slots. An absent
    quantity is ``-1``.

    Attributes
    ----------
    beam, absorbed, regions : int
        The first slot of each, or -1.
    apertures : int
        The first aperture's first slot, or -1.
    stride : int
        Slots per aperture.
    collected, amplitude : int
        Offsets within an aperture's slots, or -1.
    axis : int
        The first of the ``4 n_pol`` slots of the on-axis amplitude and its closure,
        or -1.
    n_slots : int
        Slots in all, positive.
    """

    beam: int
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
            The attributes in their declared order, int64, shape ``(9,)``.
        """
        return np.array([getattr(self, item.name) for item in fields(self)], np.int64)


def far_field_layout(
    names: Collection[str],
    *,
    n_apertures: int,
    n_regions: int,
    n_pol: int,
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
    (0, 1, 1, 3, 7)
    """
    wanted = set(names)
    collected = 0 if "aperture_spectrum" in wanted else -1
    amplitude = (
        (1 if collected == 0 else 0) if "aperture_on_axis_spectrum" in wanted else -1
    )
    stride = int(collected == 0) + (4 * n_pol if amplitude >= 0 else 0)
    counts = (
        ("beam", 1 if "beam_spectrum" in wanted else 0),
        (
            "absorbed",
            len(ABSORBED_CHANNELS) if "absorbed_power" in wanted else 0,
        ),
        ("regions", n_regions if "region_spectrum" in wanted else 0),
        ("apertures", stride * n_apertures),
        ("axis", 4 * n_pol if wanted & AXIS_FOLDED else 0),
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


def far_field_partials(problem: PropagationProblem, spec: StatsSpec) -> tuple[int, int]:
    """Return the slots and bands a compiled far-field pass holds for a problem.

    Sized from the run's own statistics at construction: their names and tables, and
    the problem's polarisations. A Cartesian or line problem whose request forms
    nothing in the pass (and any other kind) holds none.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The translated problem.
    spec : solprop.stats.StatsSpec
        The run's statistics
        (:attr:`solprop.translate.run_request.ObservationRequest.stats`).

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
    layout = far_field_layout(
        spec.names,
        n_apertures=spec.n_apertures,
        n_regions=spec.n_regions,
        n_pol=problem.shape[1],
    )
    if layout is None:
        return 0, 0
    n_x, n_y = geometry.shape
    edges = far_field_bands(n_x, n_y, layout.n_slots, problem.shape[2])
    return layout.n_slots, edges.size - 1


class FarFieldTables:
    """Hold a problem's far-field tables, formed at construction, for its statistics.

    A backend builds one when it is constructed, which forms
    :func:`solprop.translate.spatial.aperture_tables` for the run's own spec --
    refusing a geometry the far-field statistics do not admit before any step -- and
    hands :meth:`for_spec` the spec of each statistics call. The periodic channel and
    the reference backend's filtered specs carry the run's own apertures and regions,
    and get the held tables; a different spec handed to the run loop gets tables
    formed for it.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem the backend executes.
    spec : solprop.stats.StatsSpec
        The statistics of the request it was built for.

    Raises
    ------
    OutputError
        If the spec names a far-field statistic the problem's geometry does not
        admit (:func:`solprop.translate.spatial.aperture_tables`).
    """

    def __init__(self, problem: PropagationProblem, spec: StatsSpec) -> None:
        """Form the run's own tables once."""
        self._problem = problem
        self._spec = spec
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


def collect_spatial(
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
    node_chunk: int | None = None,
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
    node_chunk : int or None, optional
        The backend's node chunk, which bounds each pass's columns; ``None`` takes
        every column in one chunk.

    Returns
    -------
    dict
        Scalar or small array values, one trajectory's.

    Raises
    ------
    BackendError
        If geometry, the plane measure or a requested statistic dependency is
        missing, or a width is asked of a fluence negative beyond its rounding.

    Notes
    -----
    The far-field statistics are one pass of
    :func:`solprop.spatial.aperture_statistics` over the columns, in chunks of
    `node_chunk`, reading the plane's root through
    :func:`solprop.translate.spatial.axial_factors`, the root the stages use.
    """
    geometry = transform.geometry
    if geometry is None:
        msg = "spatial statistics require spatial geometry"
        raise BackendError(msg)
    chunk = transform.n_nodes if node_chunk is None else node_chunk
    values: dict[str, np.ndarray] = {}
    wanted = set(spec.names)
    fielded = wanted & (
        {
            "energy",
            "on_axis_spectrum",
            "on_axis_intensity",
            "peak_intensity",
            "beam_radius",
            "fwhm_r",
        }
        | set(FAR_FIELD_STATISTICS)
    )
    if fielded and flux is None:
        msg = f"spatial statistics {sorted(fielded)} need the run's plane measure"
        raise BackendError(msg)
    plane = None if flux is None else flux.at(z)
    conversion = 0.5 * EPS0 * C
    chunks = [
        slice(first, min(first + chunk, transform.n_modes))
        for first in range(0, transform.n_modes, chunk)
    ]
    axis_needed = bool(
        wanted & {"on_axis_spectrum", "on_axis_intensity", "beam_radius", "fwhm_r"}
    )
    local_needed = bool(wanted & {"peak_intensity", "beam_radius", "fwhm_r"})
    closure = None
    if plane is not None and (local_needed or axis_needed):
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
            wanted,
            grid,
            field,
            tables,
            plane=plane,
            operator=operator,
            chunk=chunk,
        )
    )
    axis_fluence = 0.0
    if axis_needed:
        assert closure is not None
        axis_values, axis_fluence = _axis_statistics(
            wanted,
            grid,
            np.einsum("r,rpw->pw", geometry.on_axis, field),
            np.einsum("r,rpw->pw", geometry.on_axis, closure),
        )
        values.update(axis_values)
    if local_needed:
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
                chunk=chunk,
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


def _local_statistics(
    wanted: set[str],
    grid: TimeGrid,
    transform: Transform,
    field: np.ndarray,
    closure: np.ndarray,
    *,
    physical: np.ndarray | None,
    axis_fluence: float,
    chunk: int,
) -> dict[str, np.ndarray]:
    """Form the peak local axial flux and the fluence widths, node chunk by chunk.

    - ``field``: one internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
    - ``closure``: its magnetic closure ``G / (eps0 c)``, same shape.
    - ``physical``: a compiled backend's borrowed physical-space field, or None.
    - ``axis_fluence``: the on-axis fluence in J/m^2, for the radial widths.
    - Returns: ``peak_intensity``, ``beam_radius`` and ``fwhm_r`` where requested.
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
    magnitudes = np.empty(transform.n_nodes)
    peak = -np.inf
    for first in range(0, transform.n_nodes, chunk):
        rows = slice(first, first + chunk)

        electric = analytic_field(grid, _rows(transform, grid, field, full, rows))
        magnetic = analytic_field(
            grid, _rows(transform, grid, closure, full_closure, rows)
        )
        intensity = conversion * np.sum(np.real(electric * np.conj(magnetic)), axis=1)
        fluence[rows] = grid.dt * np.sum(intensity, axis=-1)
        magnitudes[rows] = (
            conversion
            * grid.dt
            * np.sum(np.abs(electric) * np.abs(magnetic), axis=(1, 2))
        )
        peak = max(peak, float(np.max(intensity)))
    values: dict[str, np.ndarray] = {}
    if "peak_intensity" in wanted:
        values["peak_intensity"] = np.asarray(peak)
    if wanted & {"beam_radius", "fwhm_r"}:
        budget = fluence_budget(
            magnitudes, field.shape[1] * grid.n_t + transform.n_modes
        )
        widths = signed_fluence_widths(geometry, fluence, budget, axis_fluence)
        if "beam_radius" in wanted:
            values["beam_radius"] = np.asarray(widths[:2])
        if "fwhm_r" in wanted:
            values["fwhm_r"] = np.asarray(widths[2])
    return values


def fluence_budget(
    magnitude: Float[np.ndarray, " n_nodes"], n_terms: int
) -> Float[np.ndarray, " n_nodes"]:
    """Return the rounding budget of a signed local fluence (SEP 34 section 5.1).

    A node's fluence ``(eps0 c/2) dt sum Re(E G*)`` sums ``n_pol n_t`` products of
    two syntheses over the columns; ``n_terms`` counts both sums. A synthesis (a
    transverse FFT or a dense sum, then the temporal FFT) errs normwise: each node's
    value carries rounding of the whole field's size, which in a beam's wings far
    exceeds the value itself. Eight rounding units per term, relative to the largest
    magnitude ``(eps0 c/2) dt sum |E| |G|`` over the nodes, bound every node's error.

    Parameters
    ----------
    magnitude : numpy.ndarray
        The fluence magnitude at each node in J/m^2, or a bound on it from above: a
        backend that forms the fluence without a time transform has only the bound.
    n_terms : int
        ``n_pol * n_t + n_modes``.

    Returns
    -------
    numpy.ndarray
        The budget in J/m^2 at each node.
    """
    largest = float(np.max(magnitude)) if np.size(magnitude) else 0.0
    return np.full(np.shape(magnitude), 8 * n_terms * np.finfo(float).eps * largest)


def signed_fluence_widths(
    geometry: SpatialGeometry,
    fluence: Float[np.ndarray, " n_nodes"],
    budget: Float[np.ndarray, " n_nodes"],
    axis_fluence: float,
) -> tuple[float, float, float]:
    """Return the widths of a signed local fluence, refusing a negative profile.

    A fluence negative by more than its rounding budget (:func:`fluence_budget`) has
    no width (SEP 34 section 5.1); within the budget it is rounding and is read as
    zero.

    Parameters
    ----------
    geometry : solprop.translate.spatial.SpatialGeometry
        Radial, Cartesian, line or planar geometry.
    fluence : numpy.ndarray
        Signed fluence in J/m^2 at the nodes.
    budget : numpy.ndarray
        Its rounding budget at each node.
    axis_fluence : float
        The on-axis fluence in J/m^2, which the radial widths read.

    Returns
    -------
    tuple of float
        Second-moment, 1/e^2 and FWHM widths in m, as the geometry's width helper
        defines them.

    Raises
    ------
    BackendError
        If the fluence is negative beyond its budget anywhere.
    """
    if np.any(fluence < -budget):
        worst = int(np.argmin(fluence + budget))
        raise _negative_fluence(float(fluence[worst]), worst)
    fluence = np.maximum(fluence, 0.0)
    if geometry.kind == "radial":
        return radial_fluence_widths(
            geometry.axes[0], fluence, geometry.spatial_weights, axis_fluence
        )
    if geometry.kind == "slab":
        moment, radius, fwhm = _slab_widths(geometry, fluence)
        return moment, radius, fwhm
    return cartesian_fluence_widths(geometry.axes[0], geometry.axes[1], fluence)


def plane_fluence_widths(
    geometry: SpatialGeometry,
    moments: Float[np.ndarray, " 8"],
    n_terms: int,
    trace: Callable[..., np.ndarray],
) -> tuple[float, float, float]:
    """Return a Cartesian plane's widths from reductions a backend took of its fluence.

    :func:`signed_fluence_widths` and :func:`solprop.spatial.cartesian_fluence_widths`
    for a backend that reduces the fluence where the field lives, so that the profile
    itself never crosses: the refusal, the clamp, the centroid, the rings and the widths
    are read as those functions read them.

    Parameters
    ----------
    geometry : solprop.translate.spatial.SpatialGeometry
        The plane's ``"free"`` geometry, with at least two samples on each axis.
    moments : numpy.ndarray
        The lowest fluence in J/m^2, its node, the bound on the largest magnitude in
        J/m^2 (at least :func:`fluence_budget`'s), the largest fluence read as
        nonnegative, the count of values that aren't finite, then the sum of the fluence
        scaled by that largest value, clamped to be nonnegative, and that sum weighted
        by the samples' x and y indices.
    n_terms : int
        As for :func:`fluence_budget`.
    trace : collections.abc.Callable
        Called with the centroid in m from the axes' first samples, their spacings, the
        extent, the rings' spacing, their count, and the rings' cosines and sines. It
        returns the sum of the scaled fluence times the squared distance from the
        centroid over the squared extent, then the scaled fluence averaged on each ring.

    Returns
    -------
    tuple of float
        Second-moment, 1/e^2 and FWHM widths in m.

    Raises
    ------
    BackendError
        If the fluence is negative beyond its budget anywhere.
    GridError
        If a value isn't finite or the second moment isn't representable.
    """
    worst, node, size, maximum, nonfinite, total, moment_x, moment_y = moments
    if nonfinite:
        msg = (
            "Cartesian fluence widths require finite axes "
            "and matching nonnegative samples"
        )
        raise GridError(msg)
    if worst < -fluence_budget(np.array([size]), n_terms)[0]:
        raise _negative_fluence(float(worst), int(node))
    if maximum == 0:
        return math.nan, math.nan, math.nan
    x, y = geometry.axes
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    center_x = float(x[0]) + dx * float(moment_x / total)
    center_y = float(y[0]) + dy * float(moment_y / total)
    extent = math.hypot(
        max(center_x - x[0], x[-1] - center_x), max(center_y - y[0], y[-1] - center_y)
    )
    rings = _fluence_rings(x, y, center_x, center_y)
    empty = np.empty(0)
    packed = trace(
        center_x - float(x[0]),
        center_y - float(y[0]),
        dx,
        dy,
        extent,
        float(rings.radii[1]) if rings is not None else 0.0,
        0 if rings is None else rings.radii.size,
        empty if rings is None else rings.cosines,
        empty if rings is None else rings.sines,
    )
    values = np.ravel(packed)
    second = extent * math.sqrt(2 * float(values[0]) / float(total))
    if not math.isfinite(second):
        msg = "Cartesian fluence moments are not representable"
        raise GridError(msg)
    if rings is None:
        return second, math.nan, math.nan
    return second, *_ring_fluence_widths(rings.radii, values[1:])


def _negative_fluence(value: float, node: int) -> BackendError:
    """Return the refusal of a fluence negative beyond its rounding budget.

    - ``value``: the fluence in J/m^2.
    """
    return BackendError(
        "beam_radius and fwhm_r need a nonnegative fluence profile, and the "
        f"signed axial flux is {value:.3e} J/m^2 at node {node}, "
        "beyond its rounding budget; the width is undefined for it"
    )


def _rows(
    transform: Transform,
    grid: TimeGrid,
    source: np.ndarray,
    whole: np.ndarray | None,
    rows: slice,
) -> np.ndarray:
    """Return one node chunk of a spectrum's physical field.

    - ``source``: internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
    - ``whole``: its already synthesised physical volume, or None to synthesise the
      rows.
    - Returns: physical spectra in V/m at the chunk's nodes.
    """
    if whole is not None:
        return whole[rows]
    return (
        (transform.synthesis[rows] / grid.pad_scale)
        @ source.reshape(transform.n_modes, -1)
    ).reshape(-1, source.shape[1], source.shape[2])


def _far_field(
    names: set[str],
    grid: TimeGrid,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    tables: ApertureTables | None,
    *,
    plane: SeparableOperator | None,
    operator: SeparableOperator | None,
    chunk: int,
) -> dict[str, np.ndarray]:
    """Form the requested far-field statistics in one chunked pass.

    - ``field``: one internal spectrum, shape ``(n_x*n_y, n_pol, n_omega)``.
    - Returns: the requested far-field statistics, empty when none is asked for.
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
        """Return the plane's admittance and the operator's rates for a chunk.

        - Returns: the complex admittance and the four absorption rates.
        """
        _, rates = axial_factors(rated, grid.omega, columns)
        return axial_admittance(plane, grid.omega, columns), rates

    return aperture_statistics(
        sorted(names), grid, field, tables, axial=axial, chunk=chunk
    )


def _axis_statistics(
    wanted: set[str],
    grid: TimeGrid,
    axis: Complex[np.ndarray, "n_pol n_omega"],
    closure: Complex[np.ndarray, "n_pol n_omega"],
) -> tuple[dict[str, np.ndarray], float]:
    """Reduce the requested statistics of the on-axis spectrum.

    - ``axis``: the on-axis amplitude, the on-axis weights contracted with the spectrum,
      shape ``(n_pol, n_omega)``.
    - ``closure``: the on-axis magnetic closure ``G / (eps0 c)``, same shape.
    - Returns: requested on-axis statistics and axial fluence in J/m^2.
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

    - ``transform``: plain Cartesian axes and FFT shape, the planar slab matrices and y
      axis, or another transform kind.
    - Returns: physical coarse spectra in V/m at the nodes, or None for a dense
      transform.
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

    - ``fluence``: fluence in J/m^2 at the nodes ``ix + n_x*iy``.
    - Returns: the line widths in y (:func:`solprop.spatial.cartesian_fluence_widths` on
      one row): the moment half-width, the 1/e^2 half-width and the FWHM, in m, of the
      fluence integrated across the gap.
    """
    n_x, n_y = geometry.shape
    across = np.sum((geometry.spatial_weights * fluence).reshape(n_y, n_x), axis=1)
    return cartesian_fluence_widths(geometry.axes[1], np.zeros(1), across)

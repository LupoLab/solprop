# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read one saved plane of a run back as a coherent field plane (SEP 31).

:func:`field_plane` turns a stored plane into a
:class:`solprop.coupling.plane.FieldPlane`: its native coefficients exactly as stored,
the reference rebuilt from the run's schema-24 plane-reference record, the medium's
metric at the plane's own position, the path certificate and the provenance.
It is the extraction half of plane exchange; the manual page is
``docs/guide/workflows/coupling.md``.

A file records what it can: the frame integral, the spatial plane measure and the
identities of a general guide's reference, not the reference's arrays. A general
modal plane's physical E/H and a dense basis Gram are therefore read with the
simulation that ran it (``source=``), checked against the file's identities first
through :mod:`solprop.processing.modal`.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import cast

import numpy as np

from solprop import fourier
from solprop.adiabatic import AdiabaticAdmission
from solprop.constants import EPS0, C
from solprop.coupling import (
    FieldPlane,
    NodalField,
    PlaneBasis,
    PlaneGeometry,
    PlaneKind,
    PlaneMetric,
    PlaneProvenance,
    ScalarModeBasis,
    basis_identity,
    reference_at,
)
from solprop.exceptions import CouplingError, OutputError, ProcessingError
from solprop.grid import FreeGrid, LineGrid, RadialGrid, TimeGrid
from solprop.modes.guide import ModalGuide
from solprop.modes.protocol import Mode
from solprop.output.schema import Output, RunMetadata
from solprop.processing.axes import _plane_field, grid_of
from solprop.processing.modal import _checked_tables, _plane_rule, _spectra
from solprop.spatial import SpatialFlux, SpatialMedium
from solprop.translate.coupled import ModalMetricTables, metric_at
from solprop.translate.flux import flux_from_record
from solprop.translate.modal_tables import ModalTransformTables
from solprop.translate.spatial import axial_admittance

__all__ = ["PLANE_TOLERANCE", "field_plane"]

PLANE_TOLERANCE = 1e-9
"""Relative distance, of the larger of 1 m and abs(z), within which a z names a plane.

A requested position names a stored plane only if it is that plane to well within a
step; a nearby physical plane that was not stored is refused, not substituted.
"""

_SPACES = {
    "solprop.grid.RadialGrid": RadialGrid,
    "solprop.grid.FreeGrid": FreeGrid,
    "solprop.grid.LineGrid": LineGrid,
}
"""The transverse grids a spatial file's record names, by qualified class name."""


def _source_part(source: object, name: str) -> object:
    """Return an attribute of a ``source=`` simulation, or refuse an object without it.

    Parameters
    ----------
    source : object
        The :class:`solprop.driver.Simulation` handed in.
    name : str
        The attribute.

    Returns
    -------
    object
        Its value.

    Raises
    ------
    ProcessingError
        If `source` has no such attribute.
    """
    if not hasattr(source, name):
        msg = f"source= is a Simulation; {type(source).__name__} has no {name}"
        raise ProcessingError(msg)
    return getattr(source, name)


def _selected(output: Output, plane: int, z: float | None) -> tuple[int, float]:
    """Return the index and stored position of the plane a selection names.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output.
    plane : int
        Plane index, negative from the end.
    z : float or None
        Position that must match a stored plane, m.

    Returns
    -------
    tuple
        The index counted from the start, and its stored position in m.

    Raises
    ------
    ProcessingError
        If the index is out of range, or no stored plane is at `z`.
    """
    positions = np.asarray(output["z"], dtype=float)
    if positions.size == 0:
        msg = "the output holds no saved plane"
        raise ProcessingError(msg)
    if z is not None:
        distance = np.abs(positions - z)
        index = int(np.argmin(distance))
        if distance[index] > PLANE_TOLERANCE * max(1.0, abs(z)):
            msg = (
                f"no plane is stored at z = {z!r} m; the nearest is at "
                f"{positions[index]!r} m. Name a stored plane, or save one there"
            )
            raise ProcessingError(msg)
        return index, float(positions[index])
    index = plane + positions.size if plane < 0 else plane
    if not 0 <= index < positions.size:
        msg = f"plane {plane} is out of range for {positions.size} saved planes"
        raise ProcessingError(msg)
    return index, float(positions[index])


def _space(record: object) -> RadialGrid | FreeGrid:
    """Rebuild a spatial file's transverse grid from its configuration record.

    Parameters
    ----------
    record : object
        The ``spatial_grid`` entry: the grid itself in a memory output, its
        serialised record in a file.

    Returns
    -------
    solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The grid.

    Raises
    ------
    ProcessingError
        If the record names no known grid.
    """
    if isinstance(record, RadialGrid | FreeGrid):
        # A memory output keeps the live grid; a file keeps its serialised record.
        return record
    if not isinstance(record, Mapping):
        msg = "this spatial run's record carries no transverse grid to rebuild"
        raise ProcessingError(msg)
    fields = dict(cast("Mapping[str, object]", record))
    kind = _SPACES.get(cast("str", fields.pop("__class__", "")))
    if kind is None:
        msg = f"this run's transverse grid {record!r} is not exchanged in this release"
        raise ProcessingError(msg)
    return kind(**fields)  # pyright: ignore[reportArgumentType] - the grid's own fields


def _lineage(metadata: RunMetadata) -> tuple[str, ...]:
    """Return the configuration hashes a run descends from, oldest first.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The run's record.

    Returns
    -------
    tuple of str
        Its parent's lineage and its parent, or empty for a run launched directly.
    """
    reference = cast("Mapping[str, object]", metadata.reference)
    parent = reference.get("parent")
    if not isinstance(parent, Mapping):
        return ()
    plane = cast("Mapping[str, object]", parent).get("plane")
    if not isinstance(plane, Mapping):
        return ()
    record = cast("Mapping[str, object]", plane)
    earlier = tuple(cast("Sequence[str]", record.get("lineage", ())))
    return (*earlier, cast("str", record["source"]))


def _recorded_basis(metadata: RunMetadata, z: float) -> tuple[str, str]:
    """Return a run's route and the basis identity it recorded at a plane.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The run's record.
    z : float
        The stored plane's position, m.

    Returns
    -------
    tuple of str
        The route and the identity. A spatial or general basis is the same at every
        plane; an ordinary mode set's plane that was not planned (a natural-step
        save) has no recorded basis, and gets an identity that names it as
        unrecorded, which no port matches.
    """
    record = cast("Mapping[str, object]", metadata.reference)["basis"]
    bases = cast("Mapping[str, object]", record)
    route = cast("str", bases["route"])
    planned = np.asarray(cast("Sequence[float]", bases["z"]), dtype=float)
    identities = cast("Sequence[str | None]", bases["identity"])
    distance = np.abs(planned - z)
    index = int(np.argmin(distance))
    if route != "modal":
        index = 0
    elif distance[index] > PLANE_TOLERANCE * max(1.0, abs(z)):
        return route, basis_identity("modal", {"unrecorded at": z})
    identity = identities[index]
    return route, "" if identity is None else identity


def _spatial_metric(
    metadata: RunMetadata,
    grid: TimeGrid,
    space: RadialGrid | FreeGrid,
    z: float,
    source: object | None,
) -> tuple[PlaneMetric, SpatialMedium | None, AdiabaticAdmission | None]:
    """Return a spatial plane's admittance metric, medium and path certificate.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The run's record.
    grid : solprop.grid.TimeGrid
        Its temporal grid.
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        Its transverse grid.
    z : float
        The plane's position, m.
    source : Simulation or None
        The simulation that ran it, for its medium.

    Returns
    -------
    tuple
        The metric (with a resolved measure when the medium is at hand), the
        medium or None, and the path certificate the record rebuilds, or None.

    Raises
    ------
    ProcessingError
        If the record has no plane measure, or `source` is another run.
    """
    spatial = cast("Mapping[str, object]", metadata.spatial)
    if "flux_plan" not in spatial:
        msg = "this spatial output carries no plane measure (output schema 23)"
        raise ProcessingError(msg)
    plan = flux_from_record(cast("Mapping[str, object]", spatial["flux_plan"]))
    admittance = axial_admittance(plan.at(z), grid.omega, slice(None))
    medium = None
    flux = None
    if source is not None:
        record = cast(
            "RunMetadata",
            cast("Callable[[], object]", _source_part(source, "metadata"))(),
        )
        if record.configuration_hash() != metadata.configuration_hash():
            msg = "source= is not the simulation that wrote this output"
            raise ProcessingError(msg)
        # The simulation that wrote a spatial run always holds its bulk medium.
        medium = cast("SpatialMedium", _source_part(source, "medium"))
        flux = SpatialFlux.resolve(grid, space, medium, z=z)
    metric = PlaneMetric(
        "admittance",
        admittance=admittance,
        weights=space.spectral_weights,
        scale=0.5 * EPS0 * C,
        flux=flux,
    )
    return metric, medium, plan.admission


def _modal_profiles(
    metadata: RunMetadata, grid: TimeGrid, position: float, source: object
) -> ScalarModeBasis | None:
    """Return an ordinary modal plane's scalar profiles from its source simulation.

    Parameters
    ----------
    metadata : solprop.output.schema.RunMetadata
        The output's record.
    grid : solprop.grid.TimeGrid
        Its temporal grid.
    position : float
        The plane's position, m.
    source : Simulation
        The simulation that wrote the output.

    Returns
    -------
    solprop.coupling.ScalarModeBasis or None
        The profiles; None for a mode family with angular structure, which has
        none.

    Raises
    ------
    ProcessingError
        If `source` is not the run that wrote the output.
    """
    record = cast(
        "RunMetadata",
        cast("Callable[[], object]", _source_part(source, "metadata"))(),
    )
    if record.configuration_hash() != metadata.configuration_hash():
        msg = "source= is not the simulation that wrote this output"
        raise ProcessingError(msg)
    modes = cast("Sequence[Mode]", _source_part(source, "modes"))
    try:
        return ScalarModeBasis.from_modes(modes, grid, position)
    except CouplingError:
        return None


def _general_parts(
    output: Output,
    metadata: RunMetadata,
    index: int,
    z: float,
    source: object | None,
) -> tuple[PlaneMetric, NodalField | None]:
    """Return a general modal plane's metric and, with its source, its E/H.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The opened output.
    metadata : solprop.output.schema.RunMetadata
        Its record.
    index : int
        The plane's index.
    z : float
        The plane's position, m.
    source : Simulation or None
        The simulation that ran it.

    Returns
    -------
    tuple
        The metric -- the identity, the source's dense Gram at the plane, or
        unknown without a source -- and the nodal field or None.

    Raises
    ------
    ProcessingError
        If `source` carries no general guide or its reference is not the file's.
    """
    modal = cast("Mapping[str, object]", metadata.modal)
    declared = cast("Mapping[str, object]", modal.get("metric") or {}).get("kind")
    if source is None:
        if declared in (None, "identity"):
            return PlaneMetric("identity"), None
        return (
            PlaneMetric(
                "unknown",
                reason=(
                    f"the basis metric is {declared!r} and the file stores its "
                    "identity, not its arrays: read the plane with source="
                ),
            ),
            None,
        )
    general = _source_part(source, "general")
    guide = _source_part(source, "guide")
    if general is None or not isinstance(guide, ModalGuide):
        msg = "source= carries no general modal guide"
        raise ProcessingError(msg)
    try:
        checked = _checked_tables(
            metadata,
            cast("ModalTransformTables", _source_part(general, "tables")),
            guide,
        )
        # The E/H synthesis is electric_field's and magnetic_field's, admitted by
        # that name on every general basis.
        electric, magnetic, _ = _spectra(
            output, checked, index, None, "electric_field", guide
        )
        nodes, weights = _plane_rule(checked, guide, z)
    except OutputError as refused:
        msg = f"source= is not the reference this output was written with: {refused}"
        raise ProcessingError(msg) from refused
    kind = metadata.grid.kind
    nodal = NodalField(
        nodes,
        weights,
        fourier.spectrum_to_optics(electric, kind=kind),
        fourier.spectrum_to_optics(magnetic, kind=kind),
    )
    problem = cast("Callable[[], object]", _source_part(source, "translate"))()
    linear = _source_part(problem, "linear")
    matrix = metric_at(
        cast("ModalMetricTables | None", _source_part(linear, "modal_metric")), z
    )
    metric = (
        PlaneMetric("identity")
        if matrix is None
        else PlaneMetric("dense", matrix=np.conj(matrix))
    )
    return metric, nodal


def field_plane(
    output: Output,
    plane: int = -1,
    *,
    z: float | None = None,
    source: object | None = None,
) -> FieldPlane:
    """Read one saved plane of a run as a :class:`~solprop.coupling.plane.FieldPlane`.

    The samples are the stored optics-convention spectrum, unchanged. The reference
    is rebuilt from the run's plane-reference record (output schema 24) at the
    plane's stored position; the metric is the run's own at that position -- the
    axial admittance of a spatial plane (SEP 34), the identity for ordinary
    true-power coefficients, a general basis's Gram; the path certificate is the
    :class:`~solprop.adiabatic.AdiabaticAdmission` the spatial record rebuilds.
    Radial, Cartesian, ordinary modal and general modal runs are read; a hybrid
    planar run is refused by name, and so is a run with the modified noise model,
    whose auxiliary noise phase is not reconstructed into the plane.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened memory or HDF5 output of output schema 24.
    plane : int, optional
        Index of the saved plane, negative from the end; default the last.
    z : float or None, optional
        Position of the plane instead, m; it must be a stored plane's to
        :data:`PLANE_TOLERANCE`. A nearby plane is never substituted.
    source : solprop.driver.Simulation or None, optional
        The simulation that wrote the output, for what the file deliberately does
        not embed: a general guide's E/H reference and a dense Gram, checked against
        the file's identities; a spatial run's medium, or an ordinary mode set's
        scalar profiles (:class:`~solprop.coupling.models.ScalarModeBasis`, for a
        capillary's HE1m or a step-index fibre's HE1m modes), checked by
        configuration hash. Not needed otherwise.

    Returns
    -------
    solprop.coupling.FieldPlane
        The plane.

    Raises
    ------
    ProcessingError
        If the output predates schema 24 or holds no record, both `plane` and `z`
        are given, the selection names no stored plane, the run is planar or used
        the modified noise model, or `source` is not the run that wrote it.

    Examples
    --------
    >>> from solprop.driver import Simulation
    >>> from solprop.processing import field_plane
    >>> simulation = Simulation.from_gnlse(
    ...     0.11, 0.01, [0.0, 0.0, -1.183e-26], lambda0=835e-9,
    ...     wavelength_limits=(500e-9, 2e-6), trange=1.2e-12,
    ...     width=100e-15, peak_power=9.8, pulseshape="sech", n_save=3,
    ... )
    >>> plane = field_plane(simulation.run())
    >>> plane.kind, plane.geometry.z, plane.units
    ('modal', 0.01, 'sqrt(W)')
    """
    metadata = output.metadata
    if metadata is None:
        msg = "field_plane needs an opened output holding its run record"
        raise ProcessingError(msg)
    if metadata.schema_version < 24 or metadata.reference is None:
        msg = (
            f"this output was written at schema {metadata.schema_version} and carries "
            "no plane-reference record (output schema 24, SEP 31): its planes "
            "cannot be exchanged"
        )
        raise ProcessingError(msg)
    if metadata.noise is not None and metadata.noise.model == "modified":
        # A modified-noise run injects its one-photon background as a source term
        # and saves the field without it; the auxiliary ``noise_phase`` is what lets
        # processing restore the background. Exchanging the stored samples alone
        # would silently drop it in the next stage (SEP 31 section 4.7).
        msg = (
            "this run used the modified noise model, whose saved field leaves the "
            "one-photon background to the auxiliary noise_phase; that phase is not "
            "reconstructed into a plane, so its planes are not exchanged: rerun with "
            "noise='input' or noise=None to couple the output"
        )
        raise ProcessingError(msg)
    if z is not None and plane != -1:
        msg = f"give a plane index or a position, not both: plane={plane}, z={z}"
        raise ProcessingError(msg)
    index, position = _selected(output, plane, z)
    samples = _plane_field(output, index, None, convention="optics")
    grid = grid_of(output)
    route, identity = _recorded_basis(metadata, position)
    if route == "planar":
        msg = (
            "a hybrid planar run's columns are slab amplitudes per y frequency; "
            "planar planes are not exchanged in this release"
        )
        raise ProcessingError(msg)
    kind = cast("PlaneKind", route)
    reference = reference_at(metadata.reference, position)
    provenance = PlaneProvenance(
        metadata.configuration_hash(), index, position, _lineage(metadata), ()
    )
    geometry = PlaneGeometry(position)
    if kind in ("radial", "cartesian"):
        space = _space(metadata.configuration.get("spatial_grid"))
        basis = PlaneBasis(kind, identity, space=space)
        metric, medium, certificate = _spatial_metric(
            metadata, grid, space, position, source
        )
        return FieldPlane(
            kind,
            samples,
            grid,
            "V m" if kind == "radial" else "V/m",
            basis,
            reference,
            geometry,
            metric,
            "zero",
            medium,
            certificate,
            provenance=provenance,
        )
    labels = tuple(
        cast("str", mode.label["display_label"])
        for mode in metadata.modes
        if mode.label is not None
    )
    basis = PlaneBasis(kind, identity, labels)
    nodal = None
    profiles = None
    metric = PlaneMetric("identity")
    if kind == "general_modal":
        metric, nodal = _general_parts(output, metadata, index, position, source)
    elif source is not None:
        profiles = _modal_profiles(metadata, grid, position, source)
    return FieldPlane(
        kind,
        samples,
        grid,
        "sqrt(W)",
        basis,
        reference,
        geometry,
        metric,
        "zero",
        nodal=nodal,
        provenance=provenance,
        profiles=profiles,
    )

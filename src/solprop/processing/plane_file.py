# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Save a field plane to a file and load it back, to launch another run from it.

:func:`save_field_plane` writes what :func:`solprop.processing.planes.field_plane` reads
from a run: the samples as stored, the temporal grid's record, the basis with its
identity, the reference, the geometry, the metric's arrays, the provenance and the path
certificate. :func:`load_field_plane` rebuilds the plane, which couples as the original
does.

The file is one HDF5 file holding one plane, with its own :data:`PLANE_FILE_SCHEMA`.
A :class:`~solprop.coupling.plane.FieldPlane` lives in the physics layer, which reads no
files, so the file form is here and not on the class. A plane's live medium and its
resolved measure are Python objects with no file form, so a plane holding either is
refused: reading a run without ``source=`` gives a plane without them. The manual page
is ``docs/guide/workflows/coupling.md``.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from solprop.coupling import (
    FieldPlane,
    NodalField,
    PlaneBasis,
    PlaneGeometry,
    PlaneMetric,
    PlaneProvenance,
    PlaneReference,
    ScalarModeBasis,
    space_record,
)
from solprop.coupling.plane import AnalyticRecord
from solprop.exceptions import OutputError, ProcessingError
from solprop.output.schema import GridRecord
from solprop.processing.axes import _space_from_record, grid_from_record
from solprop.translate.flux import _admission, admission_record

__all__ = ["PLANE_FILE_SCHEMA", "load_field_plane", "save_field_plane"]

PLANE_FILE_SCHEMA = 1
"""The layout of a field-plane file this release writes and reads."""

_SCHEMA_KEY = "field_plane_schema"
_SCALARS_KEY = "scalars"


def _record(plane: FieldPlane) -> dict[str, object]:
    """Return a plane as a tree of mappings, arrays and JSON-ready values.

    - Returns: one entry per part; a part the plane does not have is None.
    """
    metric = plane.metric
    basis = plane.basis
    profiles = plane.profiles
    return {
        "kind": plane.kind,
        "samples": np.asarray(plane.samples),
        "units": plane.units,
        "exterior": plane.exterior,
        "grid": asdict(GridRecord.from_grid(plane.grid)),
        "basis": {
            "kind": basis.kind,
            "identity": basis.identity,
            "labels": list(basis.labels),
            "space": None if basis.space is None else space_record(basis.space),
        },
        "reference": plane.reference.record(),
        "geometry": asdict(plane.geometry),
        "metric": {
            "kind": metric.kind,
            "matrix": metric.matrix,
            "admittance": metric.admittance,
            "weights": metric.weights,
            "scale": metric.scale,
            "reason": metric.reason,
        },
        "certificate": None
        if plane.certificate is None
        else admission_record(plane.certificate),
        "nodal": None if plane.nodal is None else asdict(plane.nodal),
        "provenance": plane.provenance.record(),
        "profiles": None
        if profiles is None
        else {
            "family": profiles.family,
            "labels": list(profiles.labels),
            "radius": profiles.radius,
            "u": profiles.u,
            "w": profiles.w,
            "jones": profiles.jones,
            "effective_index": profiles.effective_index,
            "core_index": profiles.core_index,
            "cladding_index": profiles.cladding_index,
            "wall_term": profiles.wall_term,
            "identity": profiles.identity,
        },
        "analytic": None if plane.analytic is None else asdict(plane.analytic),
    }


def _plain(value: object) -> object:
    """Return a NumPy scalar or array as the Python value JSON writes."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    msg = f"{type(value).__name__} has no form in a field-plane file"
    raise TypeError(msg)


def _write(group: h5py.Group, record: Mapping[str, object]) -> None:
    """Write a tree: arrays as datasets, mappings as groups, the rest as JSON.

    - ``record``: the tree. A None is recorded with the scalars, so a part left out of a
      damaged file is told apart from a part the plane does not have.
    """
    scalars: dict[str, object] = {}
    for name, value in record.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(name, data=value)
        elif isinstance(value, Mapping):
            _write(group.create_group(name), cast("Mapping[str, object]", value))
        else:
            scalars[name] = value
    # JSON writes a float's shortest exact form, so the values come back bit for bit.
    group.attrs[_SCALARS_KEY] = json.dumps(scalars, default=_plain)


def _read(group: h5py.Group) -> dict[str, object]:
    """Read a tree :func:`_write` wrote."""
    record = cast(
        "dict[str, object]", json.loads(cast("str", group.attrs[_SCALARS_KEY]))
    )
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            record[name] = _read(item)
        else:
            record[name] = cast("h5py.Dataset", item)[()]
    return record


def _plane(record: Mapping[str, object]) -> FieldPlane:
    """Rebuild a plane from its tree."""

    def part(name: str) -> dict[str, object]:
        """Return one mapping of the tree."""
        return dict(cast("Mapping[str, object]", record[name]))

    basis = part("basis")
    space = basis.pop("space")
    geometry = part("geometry")
    provenance = part("provenance")
    metric = part("metric")
    nodal = record["nodal"]
    certificate = record["certificate"]
    profiles = record["profiles"]
    analytic = record["analytic"]
    return FieldPlane(
        record["kind"],  # pyright: ignore[reportArgumentType] - the constructor checks it
        cast("np.ndarray", record["samples"]),
        grid_from_record(GridRecord(**part("grid"))),  # pyright: ignore[reportArgumentType] - as asdict wrote it
        cast("str", record["units"]),
        PlaneBasis(
            cast("str", basis["kind"]),  # pyright: ignore[reportArgumentType] - checked
            cast("str", basis["identity"]),
            tuple(cast("Sequence[str]", basis["labels"])),
            None if space is None else _space_from_record(space),
        ),
        PlaneReference.from_record(part("reference")),
        PlaneGeometry(
            cast("float", geometry["z"]),
            geometry["side"],  # pyright: ignore[reportArgumentType] - checked
            tuple(cast("Sequence[float]", geometry["origin"])),  # pyright: ignore[reportArgumentType] - three
            tuple(cast("Sequence[float]", geometry["normal"])),  # pyright: ignore[reportArgumentType] - three
            tuple(cast("Sequence[float]", geometry["x_axis"])),  # pyright: ignore[reportArgumentType] - three
        ),
        PlaneMetric(
            metric["kind"],  # pyright: ignore[reportArgumentType] - checked
            metric.get("matrix"),  # pyright: ignore[reportArgumentType] - checked
            metric.get("admittance"),  # pyright: ignore[reportArgumentType] - checked
            metric.get("weights"),  # pyright: ignore[reportArgumentType] - checked
            cast("float", metric["scale"]),
            None,
            cast("str", metric["reason"]),
        ),
        record["exterior"],  # pyright: ignore[reportArgumentType] - checked
        None,
        None if certificate is None else _admission(cast("Mapping", certificate)),
        None if nodal is None else NodalField(**cast("Mapping", nodal)),
        PlaneProvenance(
            cast("str", provenance["source"]),
            cast("int | None", provenance["plane"]),
            cast("float | None", provenance["z"]),
            tuple(cast("Sequence[str]", provenance["lineage"])),
            tuple(cast("Sequence[str]", provenance["history"])),
        ),
        None
        if profiles is None
        else ScalarModeBasis(
            **{
                **cast("Mapping", profiles),
                "labels": tuple(cast("Mapping", profiles)["labels"]),
            }
        ),
        None if analytic is None else AnalyticRecord(**cast("Mapping", analytic)),
    )


def save_field_plane(plane: FieldPlane, path: Path | str) -> None:
    """Write a field plane to a new file, to launch another run from it.

    Parameters
    ----------
    plane : solprop.coupling.plane.FieldPlane
        The plane: usually :func:`~solprop.processing.planes.field_plane` of a run.
    path : pathlib.Path or str
        The file to create. An existing file is not replaced.

    Raises
    ------
    ProcessingError
        If the plane holds a live medium or a resolved measure, which have no file
        form, or is a hybrid planar plane, which is not exchanged.
    OutputError
        If the file exists or cannot be written.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from solprop.coupling import (
    ...     FieldPlane,
    ...     PlaneBasis,
    ...     PlaneGeometry,
    ...     basis_identity,
    ... )
    >>> from solprop.fourier import spectrum_to_optics
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.processing import load_field_plane, save_field_plane
    >>> from solprop.pulses import GaussianPulse
    >>> grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 200e-15)
    >>> pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    >>> plane = FieldPlane.from_samples(
    ...     spectrum_to_optics(pulse.build(grid), kind=grid.kind),
    ...     grid,
    ...     PlaneGeometry(0.0),
    ...     exterior="zero",
    ...     basis=PlaneBasis("modal", basis_identity("modal", {"modes": ["HE11"]})),
    ... )
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     path = Path(folder) / "plane.h5"
    ...     save_field_plane(plane, path)
    ...     bool((load_field_plane(path).samples == plane.samples).all())
    True
    """
    if plane.kind == "planar":
        msg = (
            "a hybrid planar plane's columns are slab amplitudes per y frequency; "
            "planar planes are not exchanged in this release"
        )
        raise ProcessingError(msg)
    if plane.medium is not None or plane.metric.flux is not None:
        msg = (
            "this plane holds a live medium, which has no file form. Read the run "
            "without source=, or give the plane no medium: a loaded plane cannot then "
            "resolve its metric again on another temporal grid"
        )
        raise ProcessingError(msg)
    record = _record(plane)
    try:
        with h5py.File(path, "x") as file:
            file.attrs[_SCHEMA_KEY] = PLANE_FILE_SCHEMA
            _write(file, record)
    except OSError as error:
        msg = f"cannot create the field-plane file {path}: {error}"
        raise OutputError(msg) from error


def load_field_plane(path: Path | str) -> FieldPlane:
    """Read a field plane back from a file :func:`save_field_plane` wrote.

    The plane has what it was saved with, its samples, grid and metric to the last bit,
    and no medium.

    Parameters
    ----------
    path : pathlib.Path or str
        The file.

    Returns
    -------
    solprop.coupling.plane.FieldPlane
        The plane.

    Raises
    ------
    OutputError
        If the file cannot be read, is not a field-plane file, declares a schema
        other than :data:`PLANE_FILE_SCHEMA`, or is damaged.
    """
    try:
        with h5py.File(path, "r") as file:
            declared = file.attrs.get(_SCHEMA_KEY)
            if declared is None:
                msg = f"{path} is not a field-plane file"
                raise OutputError(msg)
            if int(declared) != PLANE_FILE_SCHEMA:
                msg = (
                    f"{path} declares field-plane schema {declared}, and this release "
                    f"of solprop reads schema {PLANE_FILE_SCHEMA} only: read it with "
                    "the solprop version that wrote it"
                )
                raise OutputError(msg)
            record = _read(file)
        return _plane(record)
    except (OSError, KeyError, TypeError, ValueError) as error:
        msg = f"cannot read the field-plane file {path}: {error!r}"
        raise OutputError(msg) from error

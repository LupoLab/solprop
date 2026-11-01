# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Commit independently completed points to one recoverable HDF5 scan file.

Only the controller writes this file. Staged point groups are the recovery source;
layout arrays and exact CSR statistics are rebuilt from committed groups at close.
"""

import hashlib
import json
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from solprop.driver._scan_model import Scan
from solprop.exceptions import OutputError, ProcessingError
from solprop.output._scan_lock import read_scan, write_scan
from solprop.output.hdf5 import _dataset, _group
from solprop.output.schema import RunMetadata, serialise

__all__ = ["PointResult", "ScanWriter", "grid_values", "read_point", "validate_result"]


def _json(value: object) -> str:
    """Encode metadata using the single-run JSON representation.

    Parameters
    ----------
    value : object
        Value to validate or serialise.

    Returns
    -------
    str
        Canonical JSON text.
    """
    return json.dumps(serialise(value), sort_keys=True, separators=(",", ":"))


def _hash(text: str) -> str:
    """Identify a serialised common record by its SHA-256 digest.

    Parameters
    ----------
    text : str
        Canonical serialised content.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(text.encode()).hexdigest()


def _write_array(group: h5py.Group, name: str, values: np.ndarray) -> None:
    """Store numerical or textual axes without pickling objects.

    Parameters
    ----------
    group : h5py.Group
        Destination or source HDF5 group.
    name : str
        Dataset path or column name.
    values : np.ndarray
        One-dimensional axis or result array in its declared physical units.
    """
    if values.dtype.kind in "US":
        group.create_dataset(
            name, data=values.astype(object), dtype=h5py.string_dtype()
        )
    elif values.dtype.kind == "O":
        group.create_dataset(
            name,
            data=np.array([_json(v) for v in values], dtype=object),
            dtype=h5py.string_dtype(),
        )
        group[name].attrs["encoding"] = "json"
    else:
        group.create_dataset(name, data=values)


@dataclass(frozen=True)
class PointResult:
    """Carry a worker's complete result or failure to the single writer.

    Attributes
    ----------
    index : int
        Global scan index.
    status : str
        Done, failed, skipped or conflict.
    configuration_hash : str
        Single-run configuration digest, empty if construction failed.
    metadata_json : str
        Complete run metadata, empty if construction failed.
    arrays : dict
        Named retained fields, statistics and reduction arrays.
    reductions : dict
        Reducer declarations with units, parameters and source provenance.
    error : str
        Traceback or conflict explanation, empty on success.
    wall_time : float
        Build and run wall time, s.
    n_steps : int
        Attempted propagation steps.
    n_rejected : int
        Rejected propagation steps.
    boundary_monitor : collections.abc.Mapping or None
        The point's closing boundary-monitor record, without its per-plane rows
        (:meth:`solprop.monitor.BoundaryMonitor.summary`), for a done or failed point
        that was watched; ``None`` otherwise.
    """

    index: int
    status: str
    configuration_hash: str = ""
    metadata_json: str = ""
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    reductions: Mapping[str, object] = field(default_factory=dict)
    error: str = ""
    wall_time: float = 0.0
    n_steps: int = 0
    n_rejected: int = 0
    boundary_monitor: Mapping[str, object] | None = None


def _validate_metadata(result: PointResult) -> None:
    """Validate provenance even when propagation or reduction failed.

    Parameters
    ----------
    result : PointResult
        Point payload with a constructed simulation's provenance.
    """
    if len(result.configuration_hash) != 64:
        raise OutputError("completed result requires a configuration hash")
    document = json.loads(result.metadata_json)
    if not isinstance(document, dict) or "grid" not in document:
        raise OutputError("completed result requires run metadata")
    if (
        RunMetadata.from_json(result.metadata_json).configuration_hash()
        != result.configuration_hash
    ):
        raise OutputError("result metadata does not match its configuration hash")


def validate_result(result: PointResult) -> None:
    """Check the complete payload before changing committed data.

    Parameters
    ----------
    result : PointResult
        Complete point payload with status and provenance.
    """
    _validate_metadata(result)
    arrays = result.arrays
    if "z" not in arrays or arrays["z"].ndim != 1:
        raise OutputError("completed result requires a one-dimensional saved z axis")
    count = len(arrays["z"])
    for name, array in arrays.items():
        if any(part in ("", ".", "..") for part in name.split("/")):
            raise OutputError(f"invalid result array path {name!r}")
        if name.startswith("stats/") and (array.ndim < 1 or len(array) != count):
            raise OutputError(f"plane statistic {name} does not match saved planes")
        if (
            name.startswith("step_stats/")
            and name != "step_stats/counts"
            and (
                array.ndim < 1
                or "step_stats/z" not in arrays
                or len(array) != len(arrays["step_stats/z"])
            )
        ):
            raise OutputError(f"periodic statistic {name} does not match periodic rows")
    if "fields/values" in arrays and (
        "fields/planes" not in arrays
        or arrays["fields/values"].ndim != 4
        or len(arrays["fields/values"]) != len(arrays["fields/planes"])
    ):
        raise OutputError("retained fields do not match their saved-plane indices")
    for name, array in result.arrays.items():
        undefined = name.startswith(("stats/", "step_stats/")) and name.rsplit("/", 1)[
            -1
        ] in {
            "omega0",
            "fwhm_t",
            "fwhm_t_outer",
            "fwhm_r",
            "beam_radius",
            "zdw",
            "quadrature_error",
        }
        if array.dtype.kind in "fc" and (
            np.any(np.isinf(array)) or (not undefined and np.any(np.isnan(array)))
        ):
            raise OutputError(f"point {result.index}: non-finite result {name}")


class ScanWriter:
    """Own the collected file for one controller session.

    Parameters
    ----------
    path : pathlib.Path or str
        Collected HDF5 path.
    scan : Scan
        Complete scan or selection; file identity uses the complete axes.
    settings : mapping
        Retention and reducer contracts; changed contracts refuse append.
    """

    def __init__(
        self, path: Path | str, scan: Scan, settings: Mapping[str, object]
    ) -> None:
        """Initialize the declared scan component."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        description = _json(
            {
                "name": scan.name,
                "variables": scan.variables,
                "order": list(scan.variables),
                "base_seed": scan.base_seed,
                "settings": settings,
            }
        )
        # One controller at a time owns this file, and `read_scan` waits on the
        # same sidecar: taking the lock before HDF5 is opened is what makes a
        # reader in another process see a consistent file rather than a live one.
        self._session = ExitStack()
        try:
            self._session.enter_context(write_scan(self.path))
            self.file = h5py.File(self.path, "a")
            self._session.callback(self.file.close)
            if "scan_schema_version" in self.file.attrs:
                if self.file.attrs.get("identity") != _hash(description):
                    raise OutputError(
                        "scan axes, root seed or retention contract changed"
                    )
            elif len(self.file) or len(self.file.attrs):
                raise OutputError("output is not an empty or recognised scan file")
            else:
                self._initialise(scan, description)
        except BaseException:
            self._session.close()
            raise

    def _initialise(self, scan: Scan, description: str) -> None:
        """Create scheduling columns before any point is dispatched.

        Parameters
        ----------
        scan : Scan
            Cartesian scan or selection retaining global indices.
        description : str
            Serialised scan identity and retention contract.
        """
        file = self.file
        file.attrs.update(
            scan_schema_version=1,
            identity=_hash(description),
            spectral_convention="optics",
        )
        group = file.create_group("scan")
        group.attrs.update(
            name=scan.name,
            index_order="C",
            description=description,
            created=datetime.now(UTC).isoformat(),
            seed_derivation=(
                "seed axis overrides; otherwise "
                "SeedSequence([base_seed, global_index])."
                "generate_state(1, dtype=uint64)[0]; without either, noise a "
                "builder leaves unseeded takes the same with base_seed replaced by "
                "the first eight bytes of sha256(name), little-endian"
            ),
        )
        group.create_dataset("shape", data=scan.shape, dtype=np.int64)
        group.create_dataset(
            "variables", data=list(scan.variables), dtype=h5py.string_dtype()
        )
        axes = group.create_group("axes")
        for name, values in scan.variables.items():
            _write_array(axes, name, values)
        points = file.create_group("points")
        points.create_dataset("index", data=np.arange(scan.size))
        coordinates = np.array(
            [scan.point(i).multi_index for i in range(scan.size)], dtype=np.int64
        ).reshape(scan.size, len(scan.shape))
        points.create_dataset("multi_index", data=coordinates)
        for column, (name, values) in enumerate(scan.variables.items()):
            _write_array(
                points.require_group("values"), name, values[coordinates[:, column]]
            )
        for name, initial in (
            ("status", "todo"),
            ("error", ""),
            ("configuration_hash", ""),
            ("layout", ""),
            ("backend", ""),
            ("noise_seed", ""),
        ):
            points.create_dataset(
                name,
                shape=(scan.size,),
                dtype=h5py.string_dtype(),
                data=[initial] * scan.size,
            )
        points.create_dataset(
            "seed",
            data=[str(scan.point(i).seed) for i in range(scan.size)],
            dtype=h5py.string_dtype(),
        )
        for name in ("wall_time", "updated"):
            points.create_dataset(name, data=np.zeros(scan.size))
        for name in ("n_steps", "n_rejected"):
            points.create_dataset(name, data=np.zeros(scan.size, dtype=np.int64))
        file.create_group("common")
        file.create_group("staged")
        file.create_group("layouts")
        file.flush()

    def state(self, index: int) -> tuple[str, str, float]:
        """Read the durable scheduling state for a point.

        Parameters
        ----------
        index : int
            Global point index.

        Returns
        -------
        tuple
            Status, configuration hash and last ownership update in Unix seconds.
        """
        return (
            str(_dataset(self.file, "points/status").asstr()[index]),
            str(_dataset(self.file, "points/configuration_hash").asstr()[index]),
            float(_dataset(self.file, "points/updated")[index]),
        )

    def mark_running(self, index: int, now: float) -> None:
        """Publish a point claim before dispatch.

        Parameters
        ----------
        index : int
            Global point index.
        now : float
            Current Unix time, s.
        """
        _dataset(self.file, "points/status")[index] = "running"
        _dataset(self.file, "points/updated")[index] = now
        self.file.flush()

    def commit(self, result: PointResult) -> None:
        """Validate and commit a result, publishing done after its payload.

        Parameters
        ----------
        result : PointResult
            Complete worker result; failure payloads never touch numerical arrays.

        Raises
        ------
        OutputError
            If a cached configuration differs, a completed point would be demoted,
            or numerical results are not finite.
        """
        if result.status == "conflict":
            raise OutputError(self._conflict(result))
        if result.status == "skipped":
            return
        if result.status not in ("done", "failed"):
            raise OutputError(f"invalid point status {result.status!r}")
        if not 0 <= result.index < len(_dataset(self.file, "points/index")):
            raise OutputError("result index is outside the scan")
        # A committed point is the only copy of its arrays on the Local, Pool and
        # Batch routes. A failure reported for one is a failure of this attempt to
        # re-derive it -- a builder that now raises, a worker that died -- and must
        # never publish over it or delete its staged group.
        if result.status == "failed" and self.state(result.index)[0] == "done":
            raise OutputError(
                f"point {result.index} is already complete; refusing to overwrite "
                "it with a failure"
            )
        if result.status == "done":
            validate_result(result)
            self._stage(result)
        elif result.metadata_json:
            _validate_metadata(result)
            self._stage(replace(result, arrays={}))
        else:
            key = f"staged/{result.index}"
            if key in self.file:
                del self.file[key]
            for name in ("backend", "noise_seed", "layout"):
                _dataset(self.file, f"points/{name}")[result.index] = ""
        file = self.file
        for name in (
            "configuration_hash",
            "error",
            "wall_time",
            "n_steps",
            "n_rejected",
        ):
            _dataset(file, f"points/{name}")[result.index] = getattr(result, name)
        file.flush()
        _dataset(file, "points/status")[result.index] = result.status
        file.flush()

    def _conflict(self, result: PointResult) -> str:
        """Explain a refused reuse by the fields of the run record that changed.

        Parameters
        ----------
        result : PointResult
            The conflict a worker reported, with the point's new run metadata.

        Returns
        -------
        str
            The worker's explanation, followed by the changed fields -- a material
            manifest entry among them where an optmat identity, policy or version
            moved -- when both records can be read.
        """
        key = f"staged/{result.index}"
        if not result.metadata_json or key not in self.file:
            return result.error
        document = _expand(
            self.file, json.loads(str(_group(self.file, key).attrs["metadata_refs"]))
        )
        stored = RunMetadata.from_json(json.dumps(document))
        current = RunMetadata.from_json(result.metadata_json)
        return f"{result.error}; {stored.summarise_differences(current)}"

    def _stage(self, result: PointResult) -> None:
        """Write recoverable point arrays and references to common metadata.

        Parameters
        ----------
        result : PointResult
            Complete point payload with status and provenance.
        """
        file = self.file
        key = f"staged/{result.index}"
        if key in file:
            del file[key]
        point = file.create_group(key)
        document = json.loads(result.metadata_json)
        versions = document.get("versions", {})
        if isinstance(versions, dict):
            _group(file, "scan").attrs["solprop_version"] = str(
                versions.get("solprop", "unknown")
            )
        references = self._common_record(document)
        noise = document.get("noise")
        _dataset(file, "points/noise_seed")[result.index] = str(
            noise.get("seed") if isinstance(noise, dict) else None
        )
        point.attrs["metadata_refs"] = _json(references)
        point.attrs["reductions"] = _json(dict(result.reductions))
        if result.boundary_monitor is not None:
            point.attrs["boundary_monitor"] = _json(
                {
                    name: value
                    for name, value in result.boundary_monitor.items()
                    if name != "planes"
                }
            )
        units = {
            f"reduced/{name}": record.get("units")
            for name, record in result.reductions.items()
            if isinstance(record, dict)
        }
        if "fields/values" in result.arrays:
            configuration = document.get("configuration")
            per_width = (
                isinstance(configuration, dict)
                and configuration.get("transverse_measure") == "per_unit_width"
            )
            units["fields/values"] = {
                "radial": "V m",
                "free": "V/m",
                "slab": "sqrt(W/m)",
            }.get(
                document.get("simulation_kind"),
                "sqrt(W/m)" if per_width else "sqrt(W)",
            )
        point.attrs["array_units"] = _json(units)
        _dataset(file, "points/backend")[result.index] = _json(document.get("backend"))
        layout = _hash(
            _json(
                {
                    name: [
                        array.shape[1:]
                        if name.startswith("step_stats/")
                        else array.shape,
                        str(array.dtype),
                        units.get(name),
                    ]
                    for name, array in result.arrays.items()
                }
            )
        )
        _dataset(file, "points/layout")[result.index] = layout
        for name, array in result.arrays.items():
            parent, _, leaf = name.rpartition("/")
            _write_array(point.require_group(parent) if parent else point, leaf, array)

    def _common_record(self, document: object) -> object:
        # Array leaves are content-addressed before recursion: a dense synthesis
        # matrix belongs in one record, not thousands of scalar records.
        """Deduplicate metadata leaves by their content digest.

        Parameters
        ----------
        document : object
            Serialised metadata or content-addressed reference tree.

        Returns
        -------
        object
            Reference tree with leaves stored once.
        """
        if isinstance(document, dict) and "__ndarray__" not in document:
            return {key: self._common_record(value) for key, value in document.items()}
        text = _json(document)
        digest = _hash(text)
        if digest not in _group(self.file, "common"):
            _group(self.file, "common").create_dataset(
                digest, data=text, dtype=h5py.string_dtype()
            )
        return {"ref": digest}

    def record_execution(self, executor: str, policy: Mapping[str, object]) -> None:
        """Append the controller policy and measured resource calculation.

        Parameters
        ----------
        executor : str
            Executor class name.
        policy : mapping
            Effective thread, process and memory settings, including the probe.
        """
        if "script_hash" in policy:
            _group(self.file, "scan").attrs["script_hash"] = str(policy["script_hash"])
        group = self.file.require_group("executions")
        group.create_dataset(
            str(len(group)),
            data=_json(
                {
                    "executor": executor,
                    "policy": policy,
                    "recorded": datetime.now(UTC).isoformat(),
                }
            ),
            dtype=h5py.string_dtype(),
        )
        self.file.flush()

    def close(self, *, consolidate: bool = True) -> None:
        """Build layout arrays and CSR statistics, then close the writer.

        A hard interruption before this point is readable from committed staging.

        Parameters
        ----------
        consolidate : bool, optional
            False closes an unchanged collection without rebuilding its layouts.
        """
        try:
            if consolidate:
                self._collect_layouts()
        finally:
            # Closes the file, then releases the sidecar lock a waiting reader
            # in another process is blocked on.
            self._session.close()

    def _collect_layouts(self) -> None:
        """Consolidate committed points in increasing global-index order."""
        file = self.file
        done = np.flatnonzero(_dataset(file, "points/status").asstr()[...] == "done")
        by_layout: dict[str, list[int]] = {}
        for index in done:
            by_layout.setdefault(
                str(_dataset(file, "points/layout").asstr()[index]), []
            ).append(int(index))
        # Materialisations are disposable. Readers fall back to committed staging
        # until the entire replacement has been flushed.
        if "layouts" in file:
            del file["layouts"]
        layouts = file.create_group("layouts")
        for name, indices in by_layout.items():
            layout = layouts.create_group(name)
            layout.create_dataset("index", data=indices)
            first = _group(file, f"staged/{indices[0]}")
            for key in _array_paths(first):
                source = _dataset(first, key)
                if key.startswith("step_stats/"):
                    lengths = [
                        len(_dataset(file, f"staged/{index}/{key}"))
                        for index in indices
                    ]
                    offsets = np.cumsum([0, *lengths])
                    group = layout.require_group(key)
                    target = group.create_dataset(
                        "values",
                        shape=(int(offsets[-1]), *source.shape[1:]),
                        dtype=source.dtype,
                    )
                    group.create_dataset("offsets", data=offsets)
                    for row, index in enumerate(indices):
                        target[offsets[row] : offsets[row + 1]] = _dataset(
                            file, f"staged/{index}/{key}"
                        )[...]
                else:
                    parent, _, leaf = key.rpartition("/")
                    group = layout.require_group(parent) if parent else layout
                    target = group.create_dataset(
                        leaf, shape=(len(indices), *source.shape), dtype=source.dtype
                    )
                    for row, index in enumerate(indices):
                        target[row] = _dataset(file, f"staged/{index}/{key}")[...]
        self._common_fields(done)
        self._rectangular(done, next(iter(by_layout)) if len(by_layout) == 1 else None)

    def _rectangular(self, done: np.ndarray, layout_name: str | None) -> None:
        """Expose uniform arrays over the complete global point axis.

        Parameters
        ----------
        done : np.ndarray
            Completed global indices in increasing order.
        layout_name : str | None
            Uniform layout digest, or None for multiple layouts.
        """
        file = self.file
        for name in ("stats", "step_stats", "reduced", "fields", "z"):
            if name in file:
                del file[name]
        if layout_name is not None:
            layout = _group(file, f"layouts/{layout_name}")
            # Uniform rectangular datasets include incomplete points. Readers use
            # status as the validity mask; missing numerical slots are never success.
            for key in _array_paths(_group(file, f"staged/{int(done[0])}")):
                source = _dataset(file, f"staged/{int(done[0])}/{key}")
                parent, _, leaf = key.rpartition("/")
                group = file.require_group(parent) if parent else file
                if key.startswith("step_stats/"):
                    packed = group.create_group(leaf)
                    packed["values"] = _group(layout, key)["values"]
                    lengths = np.zeros(
                        len(_dataset(file, "points/index")), dtype=np.int64
                    )
                    lengths[done] = np.diff(
                        _dataset(_group(layout, key), "offsets")[...]
                    )
                    packed.create_dataset("offsets", data=np.cumsum(np.r_[0, lengths]))
                else:
                    virtual = h5py.VirtualLayout(
                        shape=(len(_dataset(file, "points/index")), *source.shape),
                        dtype=source.dtype,
                    )
                    for index in done:
                        virtual[int(index)] = h5py.VirtualSource(
                            ".", f"/staged/{int(index)}/{key}", shape=source.shape
                        )
                    target = group.create_virtual_dataset(leaf, virtual)
                    if key.startswith("reduced/"):
                        provenance = json.loads(
                            str(
                                _group(file, f"staged/{int(done[0])}").attrs[
                                    "reductions"
                                ]
                            )
                        )[leaf]
                        for attribute, value in provenance.items():
                            target.attrs[attribute] = _json(value)
        file.flush()

    def _common_fields(self, done: np.ndarray) -> None:
        """Record metadata paths that are constant or vary between points.

        Parameters
        ----------
        done : np.ndarray
            Completed global indices in increasing order.
        """
        common = _group(self.file, "common")
        if not len(done):
            return
        documents = [
            json.loads(
                str(_group(self.file, f"staged/{int(index)}").attrs["metadata_refs"])
            )
            for index in done
        ]
        leaves = [_reference_paths(document) for document in documents]
        constant = {
            key: value
            for key, value in leaves[0].items()
            if all(other.get(key) == value for other in leaves[1:])
        }
        varying = sorted(set().union(*(set(item) for item in leaves)) - constant.keys())
        common.attrs["constant_fields"] = _json(constant)
        common.attrs["varying_fields"] = _json(varying)


def _reference_paths(document: object, prefix: str = "") -> dict[str, object]:
    """Flatten reference paths for comparison without loading dense arrays.

    Parameters
    ----------
    document : object
        Serialised metadata or content-addressed reference tree.
    prefix : str
        Dotted metadata path accumulated by recursion.

    Returns
    -------
    dict
        Dotted paths and their common-record references.
    """
    if not isinstance(document, dict) or set(document) == {"ref"}:
        return {prefix: document}
    result: dict[str, object] = {}
    for key, value in document.items():
        result.update(_reference_paths(value, f"{prefix}.{key}" if prefix else key))
    return result


def _array_paths(group: h5py.Group) -> list[str]:
    """List datasets beneath a point group.

    Parameters
    ----------
    group : h5py.Group
        Destination or source HDF5 group.

    Returns
    -------
    list of str
        Dataset paths relative to the group.
    """
    paths: list[str] = []
    group.visititems(
        lambda key, value: (
            paths.append(key) if isinstance(value, h5py.Dataset) else None
        )
    )
    return paths


def _expand(file: h5py.File, value: object) -> object:
    """Resolve common metadata references in a committed point.

    Parameters
    ----------
    file : h5py.File
        Open controller-owned HDF5 file.
    value : object
        Value to validate or serialise.

    Returns
    -------
    object
        Metadata with common-record references resolved.
    """
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            return json.loads(str(_dataset(file, f"common/{value['ref']}").asstr()[()]))
        return {key: _expand(file, item) for key, item in value.items()}
    return value


def read_point(path: Path, index: int) -> Mapping[str, object]:
    """Read committed point data without exposing unfinished arrays.

    Parameters
    ----------
    path : pathlib.Path
        Scan file.
    index : int
        Global point index.

    Returns
    -------
    collections.abc.Mapping
        Status, traceback, metadata and flat array paths; for a done or failed
        point that was watched, its closing ``boundary_monitor`` record without the
        per-plane rows.

    Raises
    ------
    ProcessingError
        If the index is outside the scan.
    """
    with read_scan(path) as file:
        if not 0 <= index < len(_dataset(file, "points/index")):
            raise ProcessingError(f"scan index {index} is out of range")
        state = _dataset(file, "points/status").asstr()[index]
        result: dict[str, object] = {
            "status": state,
            "error": _dataset(file, "points/error").asstr()[index],
            "configuration_hash": _dataset(file, "points/configuration_hash").asstr()[
                index
            ],
        }
        if state in ("done", "failed") and f"staged/{index}" in file:
            group = _group(file, f"staged/{index}")
            result["metadata"] = _expand(
                file, json.loads(str(group.attrs["metadata_refs"]))
            )
            result["reductions"] = json.loads(str(group.attrs["reductions"]))
            if "boundary_monitor" in group.attrs:
                result["boundary_monitor"] = json.loads(
                    str(group.attrs["boundary_monitor"])
                )
        if state == "done":
            group = _group(file, f"staged/{index}")
            group.visititems(
                lambda key, value: (
                    result.update({key: value[...]})
                    if isinstance(value, h5py.Dataset)
                    else None
                )
            )
        return result


def grid_values(path: Path, name: str) -> np.ma.MaskedArray:
    """Read a uniform result in global-index order with unfinished points masked.

    Parameters
    ----------
    path : pathlib.Path
        Collected scan file.
    name : str
        Flat point-array path.

    Returns
    -------
    numpy.ma.MaskedArray
        Array with leading global-point axis.

    Raises
    ------
    ProcessingError
        If no completed point has the result or its shapes differ.
    """
    with read_scan(path) as file:
        states = _dataset(file, "points/status").asstr()[...]
        values = {
            int(i): _dataset(file, f"staged/{i}/{name}")[...]
            for i in np.flatnonzero(states == "done")
            if f"staged/{i}/{name}" in file
        }
        if not values:
            raise ProcessingError(f"no completed value for {name!r}")
        units = {
            json.loads(str(_group(file, f"staged/{index}").attrs["array_units"])).get(
                name
            )
            for index in values
        }
        first = next(iter(values.values()))
        if len(units) > 1 or any(
            value.shape != first.shape or value.dtype != first.dtype
            for value in values.values()
        ):
            raise ProcessingError(
                f"{name!r} has incompatible layouts; read individual points"
            )
        result = np.ma.masked_all((len(states), *first.shape), dtype=first.dtype)
        for index, value in values.items():
            result[index] = value
        return result

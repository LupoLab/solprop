# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read collected scans and declare reductions evaluated inside workers.

All numerical definitions come from the ordinary output processing functions.
Missing points are masked, never presented as successful zero-valued results.
"""

import hashlib
import inspect
import json
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import h5py
import numpy as np
from jaxtyping import Float

from solprop._display import Displayed, line, tail
from solprop.constants import C
from solprop.exceptions import ProcessingError
from solprop.output._scan_lock import read_scan
from solprop.output.capability import capability_of
from solprop.output.hdf5 import _dataset, _group
from solprop.output.scan import _expand, grid_values, read_point
from solprop.output.schema import (
    GridRecord,
    JsonValue,
    Output,
    RunMetadata,
    _rebuild,
)
from solprop.processing.axes import band_mask, grid_from_record, spectral_axis
from solprop.processing.fields import get_power, get_spectral_density
from solprop.processing.time import energy
from solprop.processing.time import peak_power as output_peak_power

__all__ = [
    "Reduction",
    "ScanOutput",
    "beam_radius",
    "custom",
    "energy_in_band",
    "energy_in_window",
    "field_at_plane",
    "mode_energies",
    "on_axis_intensity",
    "open",
    "peak_power",
    "spectrum_in_band",
]


@runtime_checkable
class Reduction(Protocol):
    """Declare a worker-side calculation and its storage contract.

    Attributes
    ----------
    name : str
        Unique dataset name.
    dtype : numpy.dtype
        Exact result dtype.
    units : str
        Physical units of the result.
    """

    @property
    def name(self) -> str:
        """Return the unique result name.

        Returns
        -------
        object
            Result of the declared operation.
        """
        ...

    @property
    def dtype(self) -> np.dtype:
        """Return the exact result dtype.

        Returns
        -------
        object
            Result of the declared operation.
        """
        ...

    @property
    def units(self) -> str:
        """Return the physical units.

        Returns
        -------
        object
            Result of the declared operation.
        """
        ...

    def shape(self, metadata: RunMetadata) -> tuple[int, ...]:
        """Declare the result dimensions.

        Parameters
        ----------
        metadata : RunMetadata
            The point's complete run record.

        Returns
        -------
        tuple of int
            Non-negative result extents, without the scan axis.
        """
        ...

    def __call__(self, output: Output) -> np.ndarray:
        """Evaluate a completed point.

        Parameters
        ----------
        output : Output
            Complete readable output, local to the worker.

        Returns
        -------
        numpy.ndarray
            Value matching the declared shape and dtype.
        """
        ...


_MODAL_OPERATIONS = frozenset({"spectrum", "band", "window", "peak", "modes"})
"""Built-in reductions whose processing definitions assume modal amplitudes."""

_STATISTIC_OPERATIONS = frozenset({"on_axis_intensity", "beam_radius"})
"""Built-in reductions of a stored statistic, which read no field."""


@dataclass(frozen=True)
class _Builtin:
    """Describe a reduction using the existing processing definitions."""

    name: str
    operation: str
    units: str
    plane: int = -1
    bounds: tuple[float, float] = (0.0, 0.0)
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))

    def shape(self, metadata: RunMetadata) -> tuple[int, ...]:
        """Return the result dimensions promised by this reduction.

        Parameters
        ----------
        metadata : RunMetadata
            Complete single-run metadata.

        Returns
        -------
        tuple of int
            Declared result dimensions.
        """
        if self.operation == "spectrum":
            return (len(_band_wavelength(metadata.grid, self.bounds)),)
        if self.operation == "field":
            explicit = metadata.configuration.get("scan_field_shape")
            if explicit is not None:
                return tuple(cast(tuple[int, ...], explicit))
            initial = metadata.configuration.get("initial")
            if isinstance(initial, np.ndarray):
                return initial.shape
            build = getattr(initial, "build", None)
            if callable(build) and metadata.spatial is None:
                return np.asarray(build(grid_from_record(metadata.grid))).shape
            raise ProcessingError("field reduction needs the input field shape")
        if self.operation == "modes":
            return (len(metadata.modes),) if len(metadata.modes) > 1 else ()
        return (2,) if self.operation == "beam_radius" else ()

    @property
    def reads_fields(self) -> bool:
        """Return whether the reduction reads its plane's field, not statistics alone.

        Returns
        -------
        bool
            False for the reductions of a statistic (``on_axis_intensity``,
            ``beam_radius``), whose plane a scan need not keep the field of.
        """
        return self.operation not in _STATISTIC_OPERATIONS

    def __call__(self, output: Output) -> np.ndarray:
        """Execute the declared worker calculation.

        Parameters
        ----------
        output : Output
            Output handler for reductions, or collected destination path for execution.

        Returns
        -------
        object
            Completed calculation or captured point failure.
        """
        return self.evaluate(output, self.plane)

    def evaluate(self, output: Output, plane: int) -> np.ndarray:
        """Evaluate the existing processing definition at one saved plane.

        Parameters
        ----------
        output : Output
            Output handler for reductions, or collected destination path for execution.
        plane : int
            Saved plane in the internal Fourier convention, or its index.

        Returns
        -------
        numpy.ndarray
            Reduction value in its declared units.

        Raises
        ------
        ProcessingError
            If a reduction defined on modal amplitudes is asked of a spatial run.
        """
        if (
            self.operation in _MODAL_OPERATIONS
            and capability_of(_metadata(output)) == "spatial"
        ):
            # The processing definitions behind these read the columns as modal
            # amplitudes in sqrt(W); a radial, Cartesian or planar column is not
            # one, and the numbers would be wrong in J.
            msg = (
                f"the {self.name} reduction measures modal amplitudes and this is a "
                f"{_metadata(output).simulation_kind} run, whose columns are not: "
                "use the run's own spatial statistics (energy, peak_intensity, "
                "on_axis_intensity, beam_radius) or a custom reduction"
            )
            raise ProcessingError(msg)
        if self.operation == "spectrum":
            return np.asarray(
                get_spectral_density(
                    output,
                    plane=plane,
                    modes="sum",
                    band="active",
                    spec_range=self.bounds,
                )[1]
            )
        if self.operation == "band":
            grid = grid_from_record(_metadata(output).grid)
            omega = grid.omega
            low, high = self.bounds

            mask = (omega >= 2 * np.pi * C / high) & (omega <= 2 * np.pi * C / low)
            return np.asarray(
                energy(
                    output, plane=plane, modes="sum", bandpass=mask.astype(np.float64)
                )
            )
        if self.operation == "window":
            t, power = get_power(output, plane=plane, modes="sum")
            mask = (t >= self.bounds[0]) & (t <= self.bounds[1])
            grid = grid_from_record(_metadata(output).grid)
            return np.asarray(np.sum(power[..., mask]) * grid.dt)
        if self.operation == "peak":
            return np.asarray(output_peak_power(output, plane=plane, modes="sum"))
        if self.operation == "modes":
            return np.asarray(energy(output, plane=plane))
        if self.operation in ("field", "on_axis_intensity", "beam_radius"):
            values = (
                output.fields()
                if self.operation == "field"
                else output[f"stats/{self.operation}"]
            )
            return np.asarray(values[plane])
        raise ProcessingError(f"unknown reduction {self.operation}")


@dataclass(frozen=True)
class _Custom:
    """Carry a callable and its explicit shape, dtype and units."""

    name: str
    function: Callable[[Output], np.ndarray]
    dimensions: tuple[int, ...]
    units: str
    dtype: np.dtype
    reads_fields: bool = True

    def shape(self, metadata: RunMetadata) -> tuple[int, ...]:
        """Return the result dimensions promised by this reduction.

        Parameters
        ----------
        metadata : RunMetadata
            Complete single-run metadata.

        Returns
        -------
        tuple of int
            Declared result dimensions.
        """
        del metadata
        return self.dimensions

    def __call__(self, output: Output) -> np.ndarray:
        """Execute the declared worker calculation.

        Parameters
        ----------
        output : Output
            Output handler for reductions, or collected destination path for execution.

        Returns
        -------
        object
            Completed calculation or captured point failure.
        """
        return self.function(output)


def _metadata(output: Output) -> RunMetadata:
    """Require the complete record needed to interpret a field.

    Parameters
    ----------
    output : Output
        Output handler for reductions, or collected destination path for execution.

    Returns
    -------
    RunMetadata
        Complete output metadata.
    """
    if output.metadata is None:
        raise ProcessingError("a reduction requires output metadata")
    return output.metadata


def _band_wavelength(
    record: GridRecord, bounds: tuple[float, float]
) -> Float[np.ndarray, " n_x"]:
    """Return the abscissa of a spectrum reduction, ascending, m.

    The bins are those :func:`~solprop.processing.fields.get_spectral_density` keeps
    with ``band='active'`` and ``spec_range=bounds``, computed by the same functions,
    so the abscissa matches the stored samples one for one.

    Parameters
    ----------
    record : GridRecord
        The point's grid record.
    bounds : tuple of float
        Closed wavelength band, m.

    Returns
    -------
    numpy.ndarray
        Wavelengths of the retained samples, ascending, m.
    """
    grid = grid_from_record(record)
    # The active band excludes the zero and Nyquist bins of a field grid, where a
    # wavelength is undefined, and the collar, where nothing physical lives.
    omega = np.sort(grid.omega[band_mask(grid, "active")])
    wavelength = spectral_axis(omega, "wavelength")[0][::-1]
    return wavelength[(wavelength >= bounds[0]) & (wavelength <= bounds[1])]


def _bounds(lo: float, hi: float, *, positive: bool) -> tuple[float, float]:
    """Validate ordered finite integration limits.

    Parameters
    ----------
    lo : float
        Finite lower limit in the reducer units.
    hi : float
        Finite upper limit in the reducer units.
    positive : bool
        Whether the lower bound must also be positive.

    Returns
    -------
    tuple of float
        Validated lower and upper limits.
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi or (positive and lo <= 0):
        raise ProcessingError("reduction bounds must be finite and increasing")
    return lo, hi


def spectrum_in_band(
    lambda_min: float, lambda_max: float, plane: int = -1
) -> Reduction:
    """Retain the total spectral energy density inside a wavelength band.

    Parameters
    ----------
    lambda_min : float
        Positive lower wavelength, m.
    lambda_max : float
        Upper wavelength greater than the lower bound, m.
    plane : int, optional
        Saved-plane index; negative indices count from the end.

    Returns
    -------
    Reduction
        Spectral energy density, J/m, summed over modes, on the ascending wavelengths
        of the run's active band that lie in the closed band; read the abscissa
        with :meth:`ScanOutput.spectrum_wavelength`.
    """
    return _Builtin(
        "spectrum_in_band",
        "spectrum",
        "J/m",
        plane,
        _bounds(lambda_min, lambda_max, positive=True),
    )


def energy_in_band(lambda_min: float, lambda_max: float, plane: int = -1) -> Reduction:
    """Measure total energy in a hard spectral band using the discrete functional.

    Parameters
    ----------
    lambda_min : float
        Positive lower wavelength, m.
    lambda_max : float
        Upper wavelength greater than the lower bound, m.
    plane : int, optional
        Saved-plane index, negative from the end.

    Returns
    -------
    Reduction
        Scalar energy, J; includes bins whose centre lies in the closed band.
    """
    return _Builtin(
        "energy_in_band",
        "band",
        "J",
        plane,
        _bounds(lambda_min, lambda_max, positive=True),
    )


def energy_in_window(t0: float, t1: float, plane: int = -1) -> Reduction:
    """Measure total energy in a closed time window on the saved time grid.

    Parameters
    ----------
    t0 : float
        Finite lower time, s.
    t1 : float
        Finite upper time greater than t0, s.
    plane : int, optional
        Saved-plane index, negative from the end.

    Returns
    -------
    Reduction
        Scalar sum of sample powers times dt, J.
    """
    return _Builtin(
        "energy_in_window", "window", "J", plane, _bounds(t0, t1, positive=False)
    )


def peak_power(plane: int = -1) -> Reduction:
    """Measure the peak of the total power across all modes.

    Parameters
    ----------
    plane : int, optional
        Saved-plane index, negative from the end.

    Returns
    -------
    Reduction
        Scalar peak power, W.
    """
    return _Builtin("peak_power", "peak", "W", plane)


def mode_energies(plane: int = -1) -> Reduction:
    """Measure each mode's energy summed over polarisation.

    Parameters
    ----------
    plane : int, optional
        Saved-plane index, negative from the end.

    Returns
    -------
    Reduction
        Mode energies, J; a single mode follows processing's scalar convention.
    """
    return _Builtin("mode_energies", "modes", "J", plane)


def field_at_plane(plane: int = -1) -> Reduction:
    """Retain a saved spectral field in the optics convention.

    Parameters
    ----------
    plane : int, optional
        Saved-plane index, negative from the end.

    Returns
    -------
    Reduction
        Complex field with its mode, polarisation and frequency axes intact.
    """
    return _Builtin(
        "field_at_plane",
        "field",
        "native field (resolved per point)",
        plane,
        dtype=np.dtype(np.complex128),
    )


def on_axis_intensity(plane: int = -1) -> Reduction:
    """Retain the existing spatial on-axis intensity statistic.

    Parameters
    ----------
    plane : int, optional
        Saved-plane index, negative from the end.

    Returns
    -------
    Reduction
        Scalar intensity, W/m²; requires the statistic in the simulation spec.
    """
    return _Builtin("on_axis_intensity", "on_axis_intensity", "W/m^2", plane)


def beam_radius(plane: int = -1) -> Reduction:
    """Retain the existing spatial beam-radius statistic.

    Parameters
    ----------
    plane : int, optional
        Saved-plane index, negative from the end.

    Returns
    -------
    Reduction
        Scalar radius, m; requires the statistic in the simulation spec.
    """
    return _Builtin("beam_radius", "beam_radius", "m", plane)


def custom(
    name: str,
    fn: Callable[[Output], np.ndarray],
    shape: tuple[int, ...],
    units: str,
    dtype: np.dtype | type = np.float64,
    *,
    reads_fields: bool = True,
) -> Reduction:
    """Wrap a custom output calculation with an explicit storage contract.

    A custom reduction is handed the point's whole run, spooled to a temporary file.
    By default that run keeps every field. One that declares ``reads_fields=False``
    reads statistics, positions and step histories only, and its spooled run keeps
    the fields the scan keeps anyway and no other (SEP 22, Q3): a field read in it is
    refused by name.

    Parameters
    ----------
    name : str
        Unique nonempty dataset name without slashes.
    fn : callable
        Worker-side function accepting a complete Output.
    shape : tuple of int
        Non-negative result extents; () denotes a scalar.
    units : str
        Physical units, or '1' for dimensionless results.
    dtype : numpy.dtype or type, optional
        Exact returned dtype, float64 by default.
    reads_fields : bool, optional
        Whether `fn` reads the run's fields; ``True``, the safe default.

    Returns
    -------
    Reduction
        Validated declaration; result shape and dtype are checked at runtime.

    Raises
    ------
    ProcessingError
        If the name or shape is invalid.
    """
    if not name or "/" in name or any(n < 0 for n in shape):
        raise ProcessingError("custom reduction needs a valid name and shape")
    return _Custom(name, fn, shape, units, np.dtype(dtype), reads_fields)


_PER_UNIT_WIDTH = {
    "J": "J/m",
    "W": "W/m",
    "J/m": "J/m per m of y",
    "sqrt(W)": "sqrt(W/m)",
}
"""A built-in reducer's units on a run uniform in y, where they are per metre of y."""


def _provenance(
    reduction: Reduction, *, metadata: RunMetadata | None = None
) -> dict[str, object]:
    """Record a reducer definition and its source availability.

    Parameters
    ----------
    reduction : Reduction
        Reduction declaration including shape, dtype and units.
    metadata : RunMetadata or None, optional
        Point metadata used to resolve modal, radial or Cartesian field units.

    Returns
    -------
    dict
        Reducer source, parameters, units and dtype.
    """
    function = reduction.function if isinstance(reduction, _Custom) else type(reduction)
    try:
        source = inspect.getsource(function)
    except OSError, TypeError:
        source = ""
    units = reduction.units
    if (
        isinstance(reduction, _Builtin)
        and reduction.operation == "field"
        and metadata is not None
    ):
        # A planar column is a slab amplitude per y frequency.
        units = {"radial": "V m", "free": "V/m", "slab": "sqrt(W/m)"}.get(
            metadata.simulation_kind, "sqrt(W)"
        )
    if (
        isinstance(reduction, _Builtin)
        and metadata is not None
        and metadata.transverse_measure == "per_unit_width"
    ):
        # A run uniform in y integrates over x per metre of y: the same numbers,
        # per unit width. Intensities are local and keep W/m^2.
        units = _PER_UNIT_WIDTH.get(units, units)
    return {
        "reducer": (
            f"{getattr(function, '__module__', '')}."
            f"{getattr(function, '__qualname__', '')}"
        ),
        "source_hash": hashlib.sha256(source.encode()).hexdigest() if source else None,
        "source_available": bool(source),
        "units": units,
        "dtype": str(reduction.dtype),
        "parameters": {"bounds": reduction.bounds, "plane": reduction.plane}
        if isinstance(reduction, _Builtin)
        else _custom_parameters(reduction)
        if isinstance(reduction, _Custom)
        else {},
        "callable_hash": _callable_hash(reduction),
    }


def _custom_parameters(reduction: _Custom) -> dict[str, object]:
    """Return a custom reduction's recorded parameters.

    Parameters
    ----------
    reduction : _Custom
        The declaration.

    Returns
    -------
    dict
        Its shape, and ``reads_fields`` where it declares that it reads none: left
        out at its default, so the provenance of every earlier reduction is unchanged.
    """
    parameters: dict[str, object] = {"shape": reduction.dimensions}
    if not reduction.reads_fields:
        parameters["reads_fields"] = False
    return parameters


def _callable_hash(reduction: Reduction) -> str | None:
    """Fingerprint serialised reducer parameters when standard pickle supports them.

    Parameters
    ----------
    reduction : Reduction
        Callable with its declared storage contract.

    Returns
    -------
    str or None
        SHA-256 of its pickle, or None for Local-only callables.
    """
    try:
        data = pickle.dumps(reduction)
    except pickle.PickleError, AttributeError, TypeError:
        return None
    return hashlib.sha256(data).hexdigest()


class ScanOutput(Displayed):
    """Read a collected scan without keeping a file handle open.

    Parameters
    ----------
    path : pathlib.Path or str
        Existing scan file.

    Attributes
    ----------
    path : pathlib.Path
        Collected file path.
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize the declared scan component."""
        self.path = Path(path)
        with read_scan(self.path) as file:
            if file.attrs.get("scan_schema_version") != 1:
                raise ProcessingError("unsupported scan file schema")

    def __repr__(self) -> str:
        """Return a one-line description naming the file, without opening it.

        Returns
        -------
        str
            For example ``ScanOutput('scan.h5')``.
        """
        return line("ScanOutput", [("", repr(self.path.name))])

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        # The file is not opened: reading a scan takes a shared lock, which blocks
        # while a controller is still writing it, and printing must never block.
        state = "present" if self.path.is_file() else "missing"
        return "ScanOutput", [
            ("file", tail(str(self.path))),
            ("state", state),
            ("contents", "read .shape, .status and .layouts, which open the file"),
        ]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the Cartesian axis lengths.

        Returns
        -------
        tuple of int
            Complete scan shape.
        """
        with read_scan(self.path) as file:
            return tuple(int(i) for i in _dataset(file, "scan/shape")[...])

    @property
    def status(self) -> np.ndarray:
        """Return point statuses in global-index order.

        Returns
        -------
        numpy.ndarray
            Strings todo, running, done or failed, shape (n_points,).
        """
        with read_scan(self.path) as file:
            return np.asarray(_dataset(file, "points/status").asstr()[...])

    @property
    def layouts(self) -> Mapping[str, tuple[int, ...]]:
        """Return known layout identifiers and their completed global points.

        Returns
        -------
        collections.abc.Mapping
            Layout digest to global indices; also works before consolidation.
        """
        result: dict[str, list[int]] = {}
        with read_scan(self.path) as file:
            for index, (state, layout) in enumerate(
                zip(
                    _dataset(file, "points/status").asstr()[...],
                    _dataset(file, "points/layout").asstr()[...],
                    strict=True,
                )
            ):
                if state == "done":
                    result.setdefault(str(layout), []).append(index)
        return {key: tuple(indices) for key, indices in result.items()}

    def point(self, index: int) -> Mapping[str, object]:
        """Read one point's arrays and provenance.

        Parameters
        ----------
        index : int
            Global point index.

        Returns
        -------
        collections.abc.Mapping
            Status, error, metadata and result groups for a completed point.

        Raises
        ------
        ProcessingError
            If the global index is out of range.
        """
        return read_point(self.path, index)

    def as_grid(self, name: str) -> np.ma.MaskedArray:
        """Reshape a uniform result over the declared variable axes.

        Parameters
        ----------
        name : str
            Result path such as 'reduced/energy_in_band' or 'stats/energy'.

        Returns
        -------
        numpy.ma.MaskedArray
            Scan shape followed by result axes; non-done points are masked.

        Raises
        ------
        ProcessingError
            If there are no successful values or their layouts differ.
        """
        values = grid_values(self.path, name)
        return values.reshape((*self.shape, *values.shape[1:]))

    def spectrum_wavelength(self, index: int | None = None) -> np.ndarray:
        """Return the wavelength abscissa of the ``spectrum_in_band`` reduction.

        The abscissa is rebuilt from a point's recorded grid and the reduction's
        recorded band, by the function that declares the reduction's shape, so it
        matches the last axis of ``as_grid("reduced/spectrum_in_band")`` sample for
        sample.

        Parameters
        ----------
        index : int or None, optional
            Global index of a completed point whose abscissa is wanted. ``None``
            returns the one abscissa shared by every completed point.

        Returns
        -------
        numpy.ndarray
            Wavelengths, ascending, m, shape ``(n_x,)``.

        Raises
        ------
        ProcessingError
            If no completed point holds the reduction, the index is out of range or
            the point does not hold it, or completed points have different abscissae.
        """
        name = "spectrum_in_band"
        with read_scan(self.path) as file:
            states = _dataset(file, "points/status").asstr()[...]
            if index is not None and not 0 <= index < len(states):
                raise ProcessingError(f"scan index {index} is out of range")
            candidates = np.flatnonzero(states == "done") if index is None else [index]
            records: dict[str, tuple[GridRecord, tuple[float, float]]] = {}
            for point in candidates:
                key = f"staged/{int(point)}"
                if f"{key}/reduced/{name}" not in file:
                    continue
                group = _group(file, key)
                parameters = json.loads(str(group.attrs["reductions"]))[name][
                    "parameters"
                ]
                if "bounds" not in parameters:
                    # A custom reducer of the same name has no band to rebuild.
                    continue
                grid = cast(
                    Mapping[str, JsonValue],
                    _expand(
                        file, json.loads(str(group.attrs["metadata_refs"]))["grid"]
                    ),
                )
                low, high = parameters["bounds"]
                bounds = (float(low), float(high))
                fingerprint = json.dumps([grid, bounds], sort_keys=True)
                records[fingerprint] = (_rebuild(GridRecord, grid), bounds)
        if not records:
            where = "" if index is None else f" at point {index}"
            raise ProcessingError(f"no completed {name!r} reduction{where}")
        abscissae = [_band_wavelength(*record) for record in records.values()]
        if any(not np.array_equal(other, abscissae[0]) for other in abscissae[1:]):
            raise ProcessingError(
                f"{name!r} has different wavelength axes between points; "
                "pass the index of the point wanted"
            )
        return abscissae[0]

    def where(self, **values: object) -> np.ndarray:
        """Find global indices whose variable values match exactly.

        Parameters
        ----------
        **values : object
            Variable names and scalar values in the original axis units.

        Returns
        -------
        numpy.ndarray
            Matching zero-based global indices, including unfinished points.

        Raises
        ------
        ProcessingError
            If a variable is unknown.
        """
        with read_scan(self.path) as file:
            mask = np.ones(len(_dataset(file, "points/index")), dtype=bool)
            for name, value in values.items():
                key = f"points/values/{name}"
                if key not in file:
                    raise ProcessingError(f"unknown scan variable {name!r}")
                dataset = _dataset(file, key)
                axis = (
                    dataset.asstr()[...]
                    if h5py.check_string_dtype(dataset.dtype)
                    else dataset[...]
                )
                mask &= axis == value
            return np.flatnonzero(mask)


def open(path: Path | str) -> ScanOutput:
    """Open a collected scan reader.

    Parameters
    ----------
    path : pathlib.Path or str
        Existing collected HDF5 scan file.

    Returns
    -------
    ScanOutput
        Reader; each operation opens and closes its own read-only handle.
    """
    return ScanOutput(path)

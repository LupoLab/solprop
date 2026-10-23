# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rebuild the axes and the layout of a saved run from its grid record.

A saved run carries a :class:`solprop.output.schema.GridRecord` -- scalars only, so that
the metadata stays small and its digest cheap -- and not the
:class:`solprop.grid.TimeGrid` it was computed on. Everything else in
:mod:`solprop.processing` needs the axes back, and :mod:`solprop.stats` needs a grid
object, so this module is the one place that rebuilds them.

Two things the rest of the package can then rely on. First, the frequency axis is stored
in the layout the kernels use: non-negative and increasing on a field grid, in FFT order
about ``omega0`` on an envelope grid (``conventions.md`` section 3). Processing shows an
**ascending** axis, and :func:`spectral_order` is the permutation that gets there.
Second, the accessors work on the grid's **active band** by default, which is what keeps
the wavelength axis finite and the selection contiguous.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Int

from solprop import fourier
from solprop.constants import C
from solprop.exceptions import ModeError, OutputError, ProcessingError
from solprop.grid import Collar, EnvelopeGrid, FieldGrid, TimeGrid
from solprop.modes.protocol import TransverseMeasure
from solprop.output.schema import Convention, GridRecord, Output, PlaneStore

__all__ = [
    "BANDS",
    "SPECTRAL_AXES",
    "Band",
    "ModeSelection",
    "SpectralAxis",
    "TransverseUnits",
    "band_mask",
    "grid_from_record",
    "grid_of",
    "layout",
    "spectral_axis",
    "spectral_order",
    "transverse_units",
]

SpectralAxis = Literal["omega", "f", "wavelength"]
"""Which spectral abscissa a density is expressed on."""

SPECTRAL_AXES: tuple[SpectralAxis, ...] = ("omega", "f", "wavelength")
"""The spectral abscissae, in SI units: rad/s, Hz and m."""

Band = Literal["active", "all"]
"""Which part of the stored spectrum an accessor returns."""

BANDS: tuple[Band, ...] = ("active", "all")
"""``'active'`` is the propagated band, ``'all'`` every stored bin."""

ModeSelection = int | Sequence[int] | Literal["sum"] | None
"""A mode index, several, the incoherent sum over all of them, or all of them."""


def grid_from_record(record: GridRecord) -> TimeGrid:
    """Rebuild a grid from the scalars a saved run records.

    Every axis and both collars come back exactly: they are functions of the recorded
    sizes, spacings, band limits and collar edges alone, evaluated by the expressions
    that built them. The grid exists so that :mod:`solprop.stats` can be called with the
    definitions it already owns.

    Parameters
    ----------
    record : solprop.output.schema.GridRecord
        The grid record of the run.

    Returns
    -------
    solprop.grid.TimeGrid
        A :class:`~solprop.grid.FieldGrid` or :class:`~solprop.grid.EnvelopeGrid`,
        validated.

    Raises
    ------
    solprop.exceptions.GridError
        If the record's scalars do not satisfy the grid invariants.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output.schema import GridRecord
    >>> from solprop.processing.axes import grid_from_record
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> rebuilt = grid_from_record(GridRecord.from_grid(grid))
    >>> rebuilt.kind, bool((rebuilt.t == grid.t).all())
    ('envelope', True)
    """
    t = (np.arange(record.n_t, dtype=float) - record.n_t / 2) * record.dt
    t_fine = (np.arange(record.n_to, dtype=float) - record.n_to / 2) * record.dt_fine
    if record.kind == "field":
        omega = np.arange(record.n_omega, dtype=float) * record.d_omega
        omega_fine = np.arange(record.n_omega_fine, dtype=float) * record.d_omega
        cls: type[TimeGrid] = FieldGrid
    else:
        # The FFT-order bin indices 0, 1, ..., n/2 - 1, -n/2, ..., -1, as `grid` builds
        # them; the coarse axis is the fine one at the bins the crop keeps.
        index = np.arange(record.n_to, dtype=float)
        index[record.n_to // 2 :] -= record.n_to
        omega_fine = index * record.d_omega + record.omega0
        omega = omega_fine[fourier.crop_index(record.n_t, record.n_to, kind="envelope")]
        cls = EnvelopeGrid
    half = 0.5 * record.n_t * record.dt
    grid = cls(
        length=record.length,
        reference_wavelength=record.reference_wavelength,
        t=t,
        omega=omega,
        t_fine=t_fine,
        omega_fine=omega_fine,
        dt=record.dt,
        dt_fine=record.dt_fine,
        ffac=record.ffac,
        omega_collar=Collar(
            0.5 * record.omega_min,
            record.omega_min,
            record.omega_max,
            record.omega_max_window,
        ),
        t_collar=Collar(-half, record.t_window_min, record.t_window_max, half),
    )
    grid.validate()
    return grid


def grid_of(output: Output) -> TimeGrid:
    """Return the grid a saved run was computed on.

    Parameters
    ----------
    output : solprop.output.schema.Output
        A memory or HDF5 output handler that has been opened.

    Returns
    -------
    solprop.grid.TimeGrid
        The rebuilt grid, as :func:`grid_from_record` returns it.

    Raises
    ------
    solprop.exceptions.OutputError
        If the handler carries no metadata, so there is no record to rebuild from.

    Examples
    --------
    >>> from solprop.processing.axes import grid_of
    >>> from solprop.processing._examples import example_output
    >>> grid_of(example_output()).kind
    'envelope'
    """
    metadata = output.metadata
    if metadata is None:
        msg = (
            "the output carries no metadata, so its grid cannot be rebuilt: open the "
            "handler with a RunMetadata before reading it"
        )
        raise OutputError(msg)
    return grid_from_record(metadata.grid)


@dataclass(frozen=True)
class TransverseUnits:
    """The units a run's transverse integrals come out in, for labels.

    Attributes
    ----------
    measure : {'area', 'per_unit_width'}
        What the run's transverse integrals measure
        (:data:`solprop.modes.protocol.TransverseMeasure`).
    power : str
        ``"W"``, or ``"W/m"`` per metre of y.
    energy : str
        ``"J"``, or ``"J/m"`` per metre of y.
    intensity : str
        ``"W/m^2"`` for both: an intensity is local.
    fluence : str
        ``"J/m^2"`` for both, for the same reason.

    Examples
    --------
    >>> from solprop.processing.axes import TransverseUnits
    >>> TransverseUnits.of("per_unit_width").energy
    'J/m'
    """

    measure: TransverseMeasure
    power: str
    energy: str
    intensity: str = "W/m^2"
    fluence: str = "J/m^2"

    @classmethod
    def of(cls, measure: TransverseMeasure) -> "TransverseUnits":  # noqa: UP037 - bound after the class body
        """Return the units of a measure.

        Parameters
        ----------
        measure : {'area', 'per_unit_width'}
            What the transverse integrals measure.

        Returns
        -------
        TransverseUnits
            The labels. The measure is already validated where it is read
            (:attr:`solprop.output.schema.RunMetadata.transverse_measure`).
        """
        if measure == "area":
            return cls(measure, "W", "J")
        return cls(measure, "W/m", "J/m")


def transverse_units(output: Output) -> TransverseUnits:
    """Return the units a saved run's powers and energies are in.

    A run on modes uniform in y -- the hollow planar guide of
    :class:`solprop.modes.slab.SlabMode` -- records that its transverse integrals are
    per metre of y (:attr:`solprop.output.schema.RunMetadata.transverse_measure`): its
    powers, energies and the processing functions that integrate them are then in
    W/m and J/m, from the same formulas. A handler without metadata is taken to
    measure an area, as every run did before the key existed.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.

    Returns
    -------
    TransverseUnits
        The unit labels.

    Examples
    --------
    >>> from solprop.processing.axes import transverse_units
    >>> from solprop.processing._examples import example_output
    >>> transverse_units(example_output()).power
    'W'
    """
    metadata = output.metadata
    return TransverseUnits.of(
        "area" if metadata is None else metadata.transverse_measure
    )


def spectral_order(grid: TimeGrid) -> Int[np.ndarray, " n_omega"]:
    """Return the permutation that sorts the stored frequency axis ascending.

    The identity on a field grid, whose axis is already increasing, and the FFT shift on
    an envelope grid, whose axis runs from the carrier up and then wraps to the bins
    below it. Applying it is what ``conventions.md`` section 3 means by "processing
    shifts for display".

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the stored spectrum lives on.

    Returns
    -------
    numpy.ndarray
        Indices into the stored axis, shape ``(n_omega,)``, such that
        ``grid.omega[order]`` increases.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.processing.axes import spectral_order
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> order = spectral_order(grid)
    >>> bool(np.all(np.diff(grid.omega[order]) > 0))
    True
    """
    if grid.kind == "field":
        return np.arange(grid.n_omega)
    return np.fft.fftshift(np.arange(grid.n_omega))


def band_mask(grid: TimeGrid, band: Band = "active") -> Bool[np.ndarray, " n_omega"]:
    """Return the mask of the bins an accessor keeps, on the stored axis.

    ``'active'`` is the propagated band ``omega_min/2 < omega < 1.1 omega_max``,
    which is the only region where the dispersion relation, the
    noise and the error norm are defined; ``'all'`` keeps every stored bin,
    including the zero and Nyquist bins of a field grid, where a wavelength is not
    defined.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the stored spectrum lives on.
    band : {'active', 'all'}
        Which part of the axis to keep.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shape ``(n_omega,)``, in storage order.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If ``band`` is not one of :data:`BANDS`.

    Examples
    --------
    >>> from solprop.grid import FieldGrid
    >>> from solprop.processing.axes import band_mask
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
    >>> int(band_mask(grid, "all").sum()) == grid.n_omega
    True
    """
    _check_choice(band, BANDS, "band")
    if band == "active":
        return grid.active_mask
    return np.ones(grid.n_omega, dtype=bool)


def spectral_axis(
    omega: Float[np.ndarray, " n_omega"], axis: SpectralAxis
) -> tuple[Float[np.ndarray, " n_omega"], Float[np.ndarray, " n_omega"]]:
    """Return a spectral abscissa and the Jacobian that maps a density onto it.

    The spectral energy density is per unit of whatever abscissa it is quoted on, so a
    change of variable carries ``|d omega / dx|``: one for angular frequency,
    ``2 pi`` for frequency and ``omega^2 / (2 pi c) = 2 pi c / lambda^2`` for
    wavelength. Units are SI: rad/s, Hz and m, so the
    densities are J/(rad/s), J/Hz and J/m.

    The wavelength abscissa **decreases** where ``omega`` increases; sorting it is the
    caller's business, because the same permutation has to be applied to the density.

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequency axis, shape ``(n_omega,)``, rad/s.
    axis : {'omega', 'f', 'wavelength'}
        The abscissa wanted.

    Returns
    -------
    x : numpy.ndarray
        The abscissa, shape ``(n_omega,)``, in rad/s, Hz or m.
    jacobian : numpy.ndarray
        The factor that multiplies a density per rad/s to give one per unit of ``x``,
        shape ``(n_omega,)``.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If ``axis`` is not one of :data:`SPECTRAL_AXES`, or if a wavelength is asked for
        on an axis that reaches zero frequency, where it is not defined.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing.axes import spectral_axis
    >>> x, jacobian = spectral_axis(np.array([2.0e15, 3.0e15]), "f")
    >>> float(x[0]) / 1e14, float(jacobian[0])
    (3.183098861837907, 6.283185307179586)
    """
    _check_choice(axis, SPECTRAL_AXES, "axis")
    if axis == "omega":
        return omega, np.ones_like(omega)
    if axis == "f":
        return omega / (2.0 * np.pi), np.full_like(omega, 2.0 * np.pi)
    if np.any(omega <= 0.0):
        msg = (
            "a wavelength axis is undefined where omega is not positive: use "
            "band='active', which excludes the zero-frequency bin"
        )
        raise ProcessingError(msg)
    return 2.0 * np.pi * C / omega, omega**2 / (2.0 * np.pi * C)


def layout(output: Output) -> tuple[int, int, int, int]:
    """Return the shape of a saved run as ``(n_z, n_modes, n_pol, n_omega)``.

    The mode and polarisation counts are not in the metadata -- a mode-averaged run
    records no modes at all -- so they are read off the stored field, which on an HDF5
    handler means reading it.

    Parameters
    ----------
    output : solprop.output.schema.Output
        A memory or HDF5 output handler that has been opened and holds at least one
        plane.

    Returns
    -------
    tuple of int
        ``(n_z, n_modes, n_pol, n_omega)``.

    Examples
    --------
    >>> from solprop.processing.axes import layout
    >>> from solprop.processing._examples import example_output
    >>> layout(example_output())[1:3]
    (1, 1)
    """
    n_z, n_modes, n_pol, n_omega = output["field"].shape
    return int(n_z), int(n_modes), int(n_pol), int(n_omega)


def _nearest_plane(output: Output, z: float) -> int:
    """Return the index of the saved plane nearest a position.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    z : float
        Position along the propagation, m.

    Returns
    -------
    int
        Index into the saved planes.

    Raises
    ------
    solprop.exceptions.OutputError
        If no plane has been saved.
    """
    positions = output["z"]
    if positions.size == 0:
        msg = "the output holds no saved plane, so no position can be selected"
        raise OutputError(msg)
    return int(np.argmin(np.abs(positions - z)))


def _plane_field(
    output: Output, plane: int | None, z: float | None, *, convention: Convention
) -> Complex[np.ndarray, "*planes n_modes n_pol n_omega"]:
    """Return the stored spectrum of the selected plane, or of every plane.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Plane index, negative from the end, or ``None``.
    z : float or None
        Plane position, m, or ``None``.
    convention : {'optics', 'fft'}
        Sign convention wanted.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_z, n_modes, n_pol, n_omega)`` when no plane was selected, and
        ``(n_modes, n_pol, n_omega)`` when one was.

    Raises
    ------
    solprop.exceptions.OutputError
        If both ``plane`` and ``z`` are given, if no plane has been saved, or if the
        index is out of range; or if the selected plane, or for no selection any
        plane, kept no field, naming the run's ``fields`` (SEP 22).
    """
    if plane is not None and z is not None:
        msg = f"give a plane index or a position, not both: plane={plane}, z={z}"
        raise OutputError(msg)
    if isinstance(output, PlaneStore) and (plane is not None or z is not None):
        # One plane is read alone, so a run that kept its field on some planes only
        # answers for those, and refuses the others by name.
        index = _nearest_plane(output, z) if z is not None else cast("int", plane)
        return output.plane_field(index, convention)
    values = output.fields(convention)
    n_z = values.shape[0]
    if n_z == 0:  # pragma: no cover - the handlers refuse to read an empty run
        msg = "the output holds no saved plane"
        raise OutputError(msg)
    if z is not None:
        return values[_nearest_plane(output, z)]
    if plane is None:
        return values
    index = plane + n_z if plane < 0 else plane
    if not 0 <= index < n_z:
        msg = f"plane {plane} is out of range for a run of {n_z} saved planes"
        raise OutputError(msg)
    return values[index]


def _keep_index(grid: TimeGrid, band: Band) -> Int[np.ndarray, " n_x"]:
    """Return the indices of the bins to return, ascending in frequency.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the stored spectrum lives on.
    band : {'active', 'all'}
        Which part of the axis to keep.

    Returns
    -------
    numpy.ndarray
        Indices into the stored axis, in order of increasing frequency.
    """
    order = spectral_order(grid)
    return order[band_mask(grid, band)[order]]


def _apply_modes(
    values: Inexact[np.ndarray, "*rest n_modes n_pol n_x"],
    modes: ModeSelection,
    *,
    allow_sum: bool,
) -> tuple[Inexact[np.ndarray, "*selected"], bool]:
    """Select or reduce the mode axis, which is the third from the end.

    Parameters
    ----------
    values : numpy.ndarray
        Array whose axes end in ``(n_modes, n_pol, n_x)``.
    modes : int, sequence of int, 'sum' or None
        The selection.
    allow_sum : bool
        Whether ``'sum'`` is meaningful for this quantity, which it is only for the ones
        that are quadratic in the field.

    Returns
    -------
    values : numpy.ndarray
        The selection, with the mode axis dropped for a single mode or a sum.
    mode_axis : bool
        Whether a mode axis is still present.

    Raises
    ------
    solprop.exceptions.ModeError
        If a sum is asked for where it is meaningless, or an index is out of range.
    """
    _check_mode_selection(modes)
    n_modes = values.shape[-3]
    if modes is None:
        return values, True
    if isinstance(modes, str):
        if not allow_sum:
            msg = (
                "modes='sum' is meaningless for a complex modal amplitude: sum a power "
                "or a spectral density instead"
            )
            raise ModeError(msg)
        return np.sum(values, axis=-3), False
    if isinstance(modes, int):
        _check_mode_index(modes, n_modes)
        return values[..., modes, :, :], False
    index = np.asarray(modes, dtype=int)
    for one in index.tolist():
        _check_mode_index(one, n_modes)
    return values[..., index, :, :], True


def _check_mode_index(mode: int, n_modes: int) -> None:
    """Reject a mode index the run does not have.

    Parameters
    ----------
    mode : int
        The index offered.
    n_modes : int
        Number of propagated modes.

    Raises
    ------
    solprop.exceptions.ModeError
        If the index is out of range.
    """
    if not -n_modes <= mode < n_modes:
        msg = f"mode {mode} is out of range for a run of {n_modes} modes"
        raise ModeError(msg)


def _squeeze(
    values: Inexact[np.ndarray, "*rest"], *, mode_axis: bool
) -> Inexact[np.ndarray, "*squeezed"]:
    """Drop the polarisation axis, and the mode axis when present, if they are trivial.

    Parameters
    ----------
    values : numpy.ndarray
        Array whose axes end in ``(n_modes?, n_pol, n_x)``.
    mode_axis : bool
        Whether a mode axis is present.

    Returns
    -------
    numpy.ndarray
        The array with trivial axes removed. The plane axis is never removed: asking for
        every plane of a one-plane run still gives a plane axis.
    """
    if values.shape[-2] == 1:
        values = values[..., 0, :]
        if mode_axis and values.shape[-2] == 1:
            values = values[..., 0, :]
    elif mode_axis and values.shape[-3] == 1:
        values = values[..., 0, :, :]
    return values


def _check_choice(value: str, allowed: tuple[str, ...], what: str) -> None:
    """Reject a keyword that is not one of a fixed set of names.

    The parameter is a plain string rather than the enumeration itself, exactly as
    :func:`solprop.fourier._check_kind` is, so that the refusal stays reachable in
    production, where no runtime type checker stands in front of it.

    Parameters
    ----------
    value : str
        The name offered.
    allowed : tuple of str
        The names accepted.
    what : str
        Name of the keyword, for the message.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the name is not one of ``allowed``.
    """
    if value not in allowed:
        msg = f"{what} must be one of {allowed}, got {value!r}"
        raise ProcessingError(msg)


def _check_mode_selection(modes: object) -> None:
    """Reject a mode selection that is none of the four forms.

    Parameters
    ----------
    modes : object
        The selection offered: an index, a sequence of indices, ``'sum'`` or ``None``.

    Raises
    ------
    solprop.exceptions.ModeError
        If it is none of those.
    """
    if modes is None or isinstance(modes, int) or modes == "sum":
        return
    if isinstance(modes, Sequence) and all(isinstance(one, int) for one in modes):
        return
    msg = f"modes must be an index, a sequence of indices, 'sum' or None, got {modes!r}"
    raise ModeError(msg)

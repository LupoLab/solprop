# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Linear optical elements applied to a pulse outside a propagation.

Every element is a **transfer function**: a callable ``H(omega)`` giving the complex
amplitude response in the optics convention of ``docs/dev/conventions.md`` section 2,
so that a spectrum is multiplied by

.. math:: H(\omega) = |H|\,e^{+i\varphi(\omega)},

the group delay is :math:`+d\varphi/d\omega`, and a positive second derivative chirps a
pulse positively, as in any textbook. A spectrum in the FFT convention needs
``exp(-i phi)`` instead; that crossing is made once, in :func:`stored_factor`, through
:func:`solprop.fourier.to_internal`. A transfer function is diagonal in frequency, so it
commutes with the half-window origin shift of a stored spectrum and the conjugation is
all the crossing needs.

The elements are :class:`TaylorPhase` (a polynomial spectral phase: GDD, TOD, ...),
:class:`Material` (a thickness of glass, or a path through a gas),
:class:`ChirpedMirror` (reflections from a dispersive mirror known by its measured
group-delay dispersion), :class:`GratingPair`, :class:`PrismPair` and :class:`Chain`.
``docs/guide/workflows/compression.md`` describes each and how the compressors in
:mod:`solprop.processing.compression` use them.
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Final, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Bool, Complex, Float
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from solprop import fourier
from solprop._display import Displayed, engineering, line, phase_text
from solprop.constants import ROOM_TEMPERATURE, C
from solprop.exceptions import MaterialError, OpticsGeometryError
from solprop.grid import TimeGrid, planck_taper
from solprop.physdata import Gas, Mixture
from solprop.physdata.glasses import GlassIndex, glass, glass_names

__all__ = [
    "Chain",
    "ChirpedMirror",
    "GratingPair",
    "Material",
    "Medium",
    "PrismGeometry",
    "PrismPair",
    "PrismPaths",
    "TaylorPhase",
    "TransferFunction",
    "brewster_angle",
    "brewster_apex",
    "chirped_mirrors",
    "littrow_angle",
    "stored_factor",
    "taylor_phase",
]

logger = logging.getLogger(__name__)

_FEMTOSECOND: Final = 1e-15
"""The mirror phase is fitted in rad/fs, which keeps the polynomial well conditioned."""


@runtime_checkable
class TransferFunction(Protocol):
    """A linear optical element, as its complex amplitude response."""

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the response at angular frequencies.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, shape ``(n_omega,)``; any sign.

        Returns
        -------
        numpy.ndarray
            ``H(omega)`` in the optics convention, shape ``(n_omega,)``. Where the
            element is not defined -- at or below zero frequency, outside a mirror's
            band -- it says what it returns.
        """
        ...


def taylor_phase(
    omega: Float[np.ndarray, " n_omega"],
    coefficients: Float[np.ndarray, " n_deg"],
    omega_ref: float,
    delay: float = 0.0,
) -> Float[np.ndarray, " n_omega"]:
    r"""Return a Taylor spectral phase, in the optics convention.

    .. math:: \varphi(\omega) = \omega t_0
       + \sum_n c_n\,(\omega-\omega_{\rm ref})^n / n!

    The one place the package writes this sum: the pulse builders and
    :class:`TaylorPhase` both call it, so a chirp put on a pulse at construction is
    removed exactly by the opposite :class:`TaylorPhase`.

    Parameters
    ----------
    omega : numpy.ndarray
        Absolute angular frequencies in rad/s, shape ``(n_omega,)``.
    coefficients : numpy.ndarray
        ``c_0, c_1, ...`` in ``s**n`` for order ``n``: carrier-envelope phase, group
        delay, group-delay dispersion, third-order dispersion and so on.
    omega_ref : float
        Frequency the expansion is about, rad/s.
    delay : float, optional
        A pure delay :math:`t_0` in s multiplying the **absolute** frequency, so that
        the whole field moves and not only its envelope; positive is later.

    Returns
    -------
    numpy.ndarray
        The phase in rad, shape ``(n_omega,)``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.optics import taylor_phase
    >>> taylor_phase(np.array([1.0, 3.0]), np.array([0.0, 0.0, 2.0]), 1.0).tolist()
    [0.0, 4.0]
    """
    detuning = omega - omega_ref
    phase = omega * delay
    for order, coefficient in enumerate(coefficients):
        phase = phase + coefficient * detuning**order / math.factorial(order)
    return np.asarray(phase)


def stored_factor(
    grid: TimeGrid, transfer: TransferFunction
) -> Complex[np.ndarray, " n_omega"]:
    """Return the factor that applies a transfer function to a stored spectrum.

    The response is evaluated on the grid's active band only and is one elsewhere: the
    bins outside it hold no field, and asking a material model for an index there would
    only raise range warnings about frequencies nothing depends on. For an element that
    blocks light, such as a mirror outside its band, this means the little that a
    spectral collar leaves outside the active band is passed rather than blocked.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the stored spectrum lives on.
    transfer : TransferFunction
        The element, in the optics convention.

    Returns
    -------
    numpy.ndarray
        The factor in the internal (FFT) convention, shape ``(n_omega,)``:
        ``conj(H)`` on the active band, by :func:`solprop.fourier.to_internal`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.optics import TaylorPhase, stored_factor
    >>> from solprop.processing._examples import example_grid
    >>> grid = example_grid()
    >>> factor = stored_factor(grid, TaylorPhase((0.0, 0.0, 100e-30), 800e-9))
    >>> bool(np.allclose(np.abs(factor), 1.0))
    True
    """
    factor = np.ones(grid.omega.shape, dtype=np.complex128)
    active = grid.active_mask
    factor[active] = fourier.to_internal(transfer(grid.omega[active]))
    return factor


@dataclass(frozen=True, eq=False)
class TaylorPhase(Displayed):
    """A polynomial spectral phase about a reference wavelength.

    Attributes
    ----------
    coefficients : sequence of float
        ``c_0, c_1, c_2, ...`` in ``s**n``: carrier-envelope phase in rad, group delay
        in s, group-delay dispersion in s^2, third-order dispersion in s^3. Zero-based,
        the same indexing as the ``phases=`` keyword of the run interfaces. A positive
        ``c_2`` chirps positively. ``c_1`` multiplies the detuning, so it delays the
        envelope and leaves the phase at the reference frequency where it was; a delay
        of the whole field, absolute frequency times a time, is a pulse's ``t0`` and the
        ``delay`` of :func:`taylor_phase`.
    wavelength : float
        Wavelength the expansion is about, m, > 0.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.optics import TaylorPhase
    >>> gdd = TaylorPhase((0.0, 0.0, 50e-30), 800e-9)
    >>> bool(np.allclose(np.abs(gdd(np.linspace(2e15, 3e15, 5))), 1.0))
    True
    """

    coefficients: Sequence[float]
    wavelength: float

    def __post_init__(self) -> None:
        """Freeze the coefficients and check the reference wavelength.

        Raises
        ------
        MaterialError
            If the wavelength is not positive or a coefficient is not finite.
        """
        values = tuple(float(value) for value in self.coefficients)
        if not all(math.isfinite(value) for value in values):
            raise MaterialError(f"Taylor coefficients must be finite, got {values}")
        if not (math.isfinite(self.wavelength) and self.wavelength > 0.0):
            raise MaterialError(
                f"the reference wavelength must be positive, got {self.wavelength!r}"
            )
        object.__setattr__(self, "coefficients", values)

    def __repr__(self) -> str:
        """Return a one-line description in the units phases are quoted in.

        Returns
        -------
        str
            For example ``TaylorPhase(GDD -250 fs^2, TOD 4000 fs^3, about 800 nm)``.
        """
        return line(
            "TaylorPhase",
            [
                ("", phase_text(self.coefficients)),
                ("", f"about {engineering(self.wavelength, 'm')}"),
            ],
        )

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the unit-modulus response.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s; any sign.

        Returns
        -------
        numpy.ndarray
            ``exp(+i phi(omega))``.
        """
        reference = 2.0 * math.pi * C / self.wavelength
        phase = taylor_phase(
            np.asarray(omega, dtype=float), np.array(self.coefficients), reference
        )
        return np.asarray(np.exp(1j * phase))


Medium = GlassIndex | Gas | str | Mixture
"""What a :class:`Material` is made of: a glass model or name, a gas, or a mixture."""


@dataclass(frozen=True, eq=False)
class Material(Displayed):
    r"""A thickness of glass, or a path through a gas.

    :math:`H = \exp\{+i[\beta(\omega) - \beta_1(\omega_0)(\omega-\omega_0)]L\}` with
    :math:`\beta = \omega\,{\rm Re}\,n/c`. The group delay at the reference wavelength
    is removed so that the pulse stays in its window; the constant :math:`\beta_0` is
    kept, so the carrier-envelope phase does move with thickness.

    Attributes
    ----------
    medium : GlassIndex, Gas, str or Mixture
        A glass model, a glass name (:func:`solprop.physdata.glasses.glass`, optmat's
        recommended model), a built-in gas (which needs `pressure`, and becomes its
        one-population :class:`~solprop.physdata.mixtures.Mixture` under optmat's
        models), or a mixture (which carries its own).
    thickness : float
        Path length in m. Negative removes material.
    wavelength : float or None
        Wavelength at which the group delay is removed, m; ``None`` keeps it, and the
        pulse leaves its window after a few millimetres of glass.
    loss : bool
        Whether to apply the absorption ``exp(-omega Im(n) L / c)``. Off by default,
        which makes the element phase-only: right for a transparent window, wrong near
        an absorption edge.
    pressure : float or None
        Gas pressure in **bar**; required for a built-in gas and refused otherwise.
    temperature : float
        Gas temperature in K; ignored for a glass and a mixture.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.optics import Material
    >>> window = Material("SiO2", 1e-3, 800e-9)
    >>> response = window(2 * np.pi * 299792458.0 / np.array([700e-9, 900e-9]))
    >>> bool(np.allclose(np.abs(response), 1.0))
    True
    """

    medium: Medium
    thickness: float
    wavelength: float | None = None
    loss: bool = False
    pressure: float | None = None
    temperature: float = ROOM_TEMPERATURE

    def __post_init__(self) -> None:
        """Resolve a glass name and check the arguments against each other.

        Raises
        ------
        MaterialError
            If the thickness is not finite, the wavelength is not positive, or the
            pressure is missing for a gas or given for anything else.
        """
        if not math.isfinite(self.thickness):
            raise MaterialError(f"thickness must be finite, got {self.thickness!r}")
        if self.wavelength is not None and not self.wavelength > 0.0:
            raise MaterialError(
                f"the reference wavelength must be positive, got {self.wavelength!r}"
            )
        medium = self.medium
        named_glass = (
            isinstance(medium, str)
            and not isinstance(medium, Gas)
            and medium.strip().lower() in glass_names()
        )
        if named_glass:
            medium = glass(str(medium))
            object.__setattr__(self, "medium", medium)
        is_gas = isinstance(medium, (Gas, str))
        if is_gas and self.pressure is None:
            raise MaterialError(f"a path through {medium!s} needs its pressure in bar")
        if not is_gas and self.pressure is not None:
            raise MaterialError("pressure is for a built-in gas only; omit it")
        if is_gas:
            # A built-in gas is optmat's (SEP 3): its one-population fill at the
            # pressure and temperature given, whose index the mixture path takes.
            object.__setattr__(
                self,
                "medium",
                Mixture.fill(
                    {cast("Gas | str", medium): 1.0},
                    float(cast("float", self.pressure)),
                    temperature=self.temperature,
                ),
            )

    def _medium_text(self) -> str:
        """Return the medium as text, for printing.

        Returns
        -------
        str
            A glass by name and source, a gas with its pressure, a mixture by its repr.
        """
        medium = self.medium
        if isinstance(medium, GlassIndex):
            return f"{medium.name} ({medium.source})"
        if self.pressure is not None and isinstance(medium, Mixture):
            return f"{'+'.join(medium.labels)} {self.pressure:g} bar"
        return repr(medium)

    def __repr__(self) -> str:
        """Return a one-line description: how much of what.

        Returns
        -------
        str
            For example ``Material(1.5 mm of SiO2 (fit/SiO2), delay removed at
            800 nm)``.
        """
        fields: list[tuple[str, object]] = [
            ("", f"{engineering(self.thickness, 'm')} of {self._medium_text()}")
        ]
        if self.wavelength is not None:
            fields.append(("", f"delay removed at {engineering(self.wavelength, 'm')}"))
        if self.loss:
            fields.append(("", "with absorption"))
        return line("Material", fields)

    def _index(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Return the index and its analytic frequency derivative.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, > 0.

        Returns
        -------
        tuple of numpy.ndarray
            ``n`` and ``dn/domega`` in s/rad, complex.
        """
        medium = self.medium
        if isinstance(medium, GlassIndex):
            return (
                np.asarray(medium.index(omega), dtype=complex),
                np.asarray(medium.dindex_domega(omega), dtype=complex),
            )
        # Construction resolved a gas name to its mixture.
        fill = cast("Mixture", medium)
        here = np.array([0.0])
        index = np.sqrt(1.0 + fill.susceptibility(omega, here)[0].astype(complex))
        slope = fill.dsusceptibility_domega(omega, here)[0] / (2.0 * index)
        return index, slope

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the response of the path.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.

        Returns
        -------
        numpy.ndarray
            ``H(omega)``; one at and below zero frequency, where there is no index,
            and one wherever the model returns a value that is not finite -- exactly on
            a Sellmeier pole, say. Inside a model's validity range neither happens.

        Warns
        -----
        MaterialRangeWarning
            If a frequency lies outside the validity range of the index model.
        """
        values = np.asarray(omega, dtype=float)
        response = np.ones(values.shape, dtype=np.complex128)
        positive = values > 0.0
        if not np.any(positive):
            return response
        used = values[positive]
        index, _ = self._index(used)
        phase = used * np.real(index) / C
        if self.wavelength is not None:
            reference = np.array([2.0 * math.pi * C / self.wavelength])
            at_reference, slope = self._index(reference)
            # beta1 from the model's own analytic dn/domega, never a finite difference.
            delay = float(np.real(at_reference[0] + reference[0] * slope[0])) / C
            phase = phase - delay * (used - reference[0])
        factor = np.exp(1j * phase * self.thickness)
        if self.loss:
            factor = factor * np.exp(-used * np.imag(index) * self.thickness / C)
        response[positive] = np.where(np.isfinite(factor), factor, 1.0)
        return response


# ---------------------------------------------------------------------------- mirrors


@dataclass(frozen=True)
class _MirrorRecord:
    """How one shipped data set is read: reference wavelength, fit and band.

    Attributes
    ----------
    files : tuple of str
        Resource names: reflectivity then dispersion, or one file holding both.
    delimiter : str or None
        Column separator; ``None`` is whitespace.
    skip : int
        Header lines after the ``#`` provenance block.
    reflectivity : tuple of int or None
        Columns (wavelength, value...) of the reflectivity in percent; several value
        columns are combined by their geometric mean. ``None``: no reflectivity data.
    dispersion : tuple of int
        Columns (wavelength, value...) of the dispersion; several are averaged.
    group_delay : bool
        Whether the dispersion columns hold group delay in fs rather than GDD in fs^2.
    reference : float
        Wavelength about which the constant and linear phase are removed, m.
    fit_order : int
        Order of the polynomial fitted to the phase for that removal.
    fit_window : tuple of float or None
        Angular-frequency window of the fit in rad/s; ``None`` is all the data.
    band : tuple of float
        The four Planck-taper wavelengths in m: zero, one, one, zero.
    """

    files: tuple[str, ...]
    delimiter: str | None
    skip: int
    reflectivity: tuple[int, ...] | None
    dispersion: tuple[int, ...]
    group_delay: bool
    reference: float
    fit_order: int
    fit_window: tuple[float, float] | None
    band: tuple[float, float, float, float]


_VISIBLE_FIT: Final = (2e15, 4e15)
"""The fit window for PC70 and the Thorlabs set, 471 to 942 nm, chosen because large
kinks at the edge of the frequency window confuse the fit."""

_MIRRORS: Final = {
    "PC70": _MirrorRecord(
        ("PC70_R.csv", "PC70_GDD.csv"),
        ",",
        1,
        (0, 1, 2),
        (0, 1, 2),
        False,
        800e-9,
        5,
        _VISIBLE_FIT,
        (400e-9, 450e-9, 1200e-9, 1300e-9),
    ),
    "HD59": _MirrorRecord(
        ("HD59_GDD.dat",),
        None,
        2,
        None,
        (0, 1),
        False,
        1030e-9,
        5,
        None,
        (993e-9, 1000e-9, 1060e-9, 1075e-9),
    ),
    "PC147": _MirrorRecord(
        ("PC147.txt",),
        "\t",
        1,
        (0, 1),
        (2, 3),
        False,
        1030e-9,
        5,
        None,
        (640e-9, 650e-9, 1350e-9, 1360e-9),
    ),
    "PC1611": _MirrorRecord(
        ("PC1611.txt",),
        "\t",
        1,
        (0, 1),
        (2, 3),
        False,
        1030e-9,
        5,
        None,
        (850e-9, 855e-9, 1195e-9, 1200e-9),
    ),
    "PC1821": _MirrorRecord(
        ("PC1821.txt",),
        "\t",
        1,
        (0, 1),
        (2, 3),
        False,
        1030e-9,
        5,
        None,
        (800e-9, 805e-9, 1345e-9, 1350e-9),
    ),
    "HD120": _MirrorRecord(
        ("HD120.csv",),
        ",",
        1,
        (0, 1),
        (2, 3),
        False,
        1030e-9,
        5,
        None,
        (880e-9, 900e-9, 1200e-9, 1220e-9),
    ),
    "ThorlabsUMC": _MirrorRecord(
        ("UCxx-15FS_R.csv", "UCxx-15FS_GD.csv"),
        ",",
        1,
        (0, 2),
        (0, 2),
        True,
        800e-9,
        3,
        _VISIBLE_FIT,
        (640e-9, 650e-9, 1050e-9, 1100e-9),
    ),
}


def chirped_mirrors() -> tuple[str, ...]:
    """Return the names of the shipped chirped-mirror data sets.

    Returns
    -------
    tuple of str
        Names :meth:`ChirpedMirror.lookup` accepts. Where the data sets came from is
        not recorded (``docs/guide/workflows/compression.md``).

    Examples
    --------
    >>> from solprop.optics import chirped_mirrors
    >>> "PC70" in chirped_mirrors()
    True
    """
    return tuple(_MIRRORS)


def _parse_columns(
    text: str, name: str, record: _MirrorRecord, columns: tuple[int, ...]
) -> Float[np.ndarray, "rows columns"]:
    """Return chosen numeric columns of a mirror data file's text.

    Parameters
    ----------
    text : str
        The file's content.
    name : str
        The file's name, for the refusal.
    record : _MirrorRecord
        How the file is laid out.
    columns : tuple of int
        The columns wanted. A vendor table may leave cells of *other* columns empty.

    Returns
    -------
    numpy.ndarray
        The chosen columns, rows by ``len(columns)``.

    Raises
    ------
    MaterialError
        If a wanted cell is missing or is not a number.
    """
    lines = [
        row for row in text.splitlines() if row.strip() and not row.startswith("#")
    ][record.skip :]
    rows = []
    for row in lines:
        cells = row.split(record.delimiter)
        try:
            rows.append([float(cells[column]) for column in columns])
        except (IndexError, ValueError) as error:
            raise MaterialError(f"{name}: unreadable data row {row!r}") from error
    return np.array(rows, dtype=float)


def _read_columns(
    name: str, record: _MirrorRecord, columns: tuple[int, ...]
) -> Float[np.ndarray, "rows columns"]:
    """Return chosen numeric columns of one shipped mirror file.

    Parameters
    ----------
    name : str
        Resource name under ``solprop/physdata/data/mirrors``.
    record : _MirrorRecord
        How the file is laid out.
    columns : tuple of int
        The columns wanted.

    Returns
    -------
    numpy.ndarray
        The chosen columns, rows by ``len(columns)``.
    """
    source = resources.files("solprop.physdata") / "data" / "mirrors" / name
    return _parse_columns(source.read_text(encoding="utf-8"), name, record, columns)


def _mean_columns(
    table: Float[np.ndarray, "rows columns"], *, geometric: bool
) -> tuple[Float[np.ndarray, " rows"], Float[np.ndarray, " rows"]]:
    """Return a wavelength column and the mean of the value columns beside it.

    Parameters
    ----------
    table : numpy.ndarray
        The wavelength in nm, then one or more value columns.
    geometric : bool
        Geometric mean (reflectivities) rather than arithmetic (dispersions).

    Returns
    -------
    tuple of numpy.ndarray
        Wavelength in m and the mean value, in the file's units.
    """
    values = table[:, 1:]
    if geometric:
        # A root of the product rather than the exponential of a mean logarithm, so
        # that a reflectivity of exactly zero is a zero and not a warning.
        return table[:, 0] * 1e-9, np.prod(values, axis=1) ** (1.0 / values.shape[1])
    return table[:, 0] * 1e-9, np.mean(values, axis=1)


@dataclass(frozen=True, eq=False)
class ChirpedMirror(Displayed):
    r"""Reflections from a dispersive mirror known by its measured dispersion.

    One reflection multiplies the spectrum by
    :math:`\sqrt{R(\omega)}\,e^{+i\varphi(\omega)}\,W(\omega)`, where :math:`\varphi`
    is the double integral of the group-delay dispersion over angular frequency with
    its constant and linear parts removed (an arbitrary phase and an overall delay),
    and :math:`W` is a Planck taper to zero outside the band the data are trusted on.
    Build one with :meth:`from_data` or :meth:`lookup`; choose the number of bounces
    with :meth:`bounces`.

    Attributes
    ----------
    omega : numpy.ndarray
        Angular frequencies of the phase samples in rad/s, ascending.
    phase : numpy.ndarray
        Phase per reflection in rad on `omega`.
    reflectivity_omega : numpy.ndarray
        Angular frequencies of the reflectivity samples, ascending; empty for a
        phase-only mirror.
    amplitude : numpy.ndarray
        Amplitude reflectivity ``sqrt(R)`` on `reflectivity_omega`.
    band : tuple of float
        The taper's four wavelengths in m: zero, one, one, zero.
    reflections : int
        Number of reflections. Negative is the phase-only inverse of that many: the
        amplitude is left alone inside the band and, like the forward mirror, zero
        outside it.
    name : str
        A label for printing.

    Examples
    --------
    >>> from solprop.optics import ChirpedMirror
    >>> ChirpedMirror.lookup("PC70").bounces(4).reflections
    4
    """

    omega: Float[np.ndarray, " n_phase"]
    phase: Float[np.ndarray, " n_phase"]
    reflectivity_omega: Float[np.ndarray, " n_reflectivity"]
    amplitude: Float[np.ndarray, " n_reflectivity"]
    band: tuple[float, float, float, float]
    reflections: int = 1
    name: str = ""
    _phase_spline: CubicSpline = field(init=False, repr=False)
    _amplitude_spline: CubicSpline | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the phase interpolant.

        Raises
        ------
        MaterialError
            If the phase samples are not ascending in frequency or too few.
        """
        if self.omega.size < 4 or np.any(np.diff(self.omega) <= 0.0):
            raise MaterialError(
                "a chirped mirror needs at least four phase samples on an ascending "
                "frequency axis"
            )
        object.__setattr__(
            self,
            "_phase_spline",
            CubicSpline(self.omega, self.phase, extrapolate=False),
        )
        amplitude = None
        if self.amplitude.size:
            wavelength = 2.0 * math.pi * C / self.reflectivity_omega
            amplitude = CubicSpline(
                wavelength[::-1], self.amplitude[::-1], extrapolate=False
            )
        object.__setattr__(self, "_amplitude_spline", amplitude)

    def __repr__(self) -> str:
        """Return a one-line description: which mirror, how many times, over what.

        Returns
        -------
        str
            For example ``ChirpedMirror(PC70, 4 reflections, 450-1200 nm)``.
        """
        low, high = self.band[1], self.band[2]
        return line(
            "ChirpedMirror",
            [
                ("", self.name or "from data"),
                ("", f"{self.reflections} reflections"),
                ("", f"{low * 1e9:.4g}-{high * 1e9:.4g} nm"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        low, high = self.band[1], self.band[2]
        centre = np.array([4.0 * math.pi * C / (low + high)])
        amplitude = (
            "phase only"
            if not self.amplitude.size
            else (
                f"{float(np.min(self.amplitude) ** 2):.4g} to "
                f"{float(np.max(self.amplitude) ** 2):.4g} (power, over the data)"
            )
        )
        return f"ChirpedMirror {self.name or 'from data'}", [
            ("reflections", self.reflections),
            ("band", f"{low * 1e9:.4g}-{high * 1e9:.4g} nm, tapered to zero outside"),
            ("GDD at band centre", f"{float(self.gdd(centre)[0]) * 1e30:.4g} fs^2"),
            ("reflectivity", amplitude),
            ("phase samples", self.omega.size),
        ]

    @classmethod
    def from_data(  # noqa: PLR0913 - two data columns, their axis and how to reduce them
        cls,
        wavelength: Float[np.ndarray, " n_wavelength"],
        reflectivity: Float[np.ndarray, " n_reflectivity"] | None,
        *,
        gdd: Float[np.ndarray, " n_gdd"] | None = None,
        group_delay: Float[np.ndarray, " n_delay"] | None = None,
        reference_wavelength: float,
        band: tuple[float, float],
        taper_width: float = 20e-9,
        fit_order: int = 5,
        fit_window: tuple[float, float] | None = None,
        name: str = "",
    ) -> "ChirpedMirror":  # noqa: UP037 - beartype evaluates the annotation eagerly
        """Build a mirror from a data sheet's columns.

        Parameters
        ----------
        wavelength : numpy.ndarray
            Wavelengths of the samples in m, strictly monotonic.
        reflectivity : numpy.ndarray or None
            **Power** reflectivity per reflection in [0, 1] on `wavelength`, or ``None``
            for a phase-only mirror.
        gdd : numpy.ndarray or None, optional
            Group-delay dispersion per reflection in s^2. Exactly one of this and
            `group_delay`.
        group_delay : numpy.ndarray or None, optional
            Group delay per reflection in s.
        reference_wavelength : float
            Wavelength about which the constant and linear phase are removed, m.
        band : tuple of float
            Shortest and longest wavelength in m the mirror is used over.
        taper_width : float, optional
            Width in m of the taper to zero outside `band`.
        fit_order : int, optional
            Order of the polynomial fitted to the integrated phase, of which only the
            constant and linear terms are subtracted.
        fit_window : tuple of float or None, optional
            Angular-frequency window of that fit in rad/s; ``None`` is all the data.
        name : str, optional
            A label.

        Returns
        -------
        ChirpedMirror
            One reflection; see :meth:`bounces`.

        Raises
        ------
        MaterialError
            If both or neither dispersion column is given, a reflectivity lies outside
            [0, 1], or the wavelength axis is not monotonic.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.optics import ChirpedMirror
        >>> wavelength = np.linspace(600e-9, 1000e-9, 401)
        >>> mirror = ChirpedMirror.from_data(
        ...     wavelength, None, gdd=np.full(401, -50e-30),
        ...     reference_wavelength=800e-9, band=(650e-9, 950e-9))
        >>> f"{mirror.gdd(np.array([2.355e15]))[0] * 1e30:.1f}"
        '-50.0'
        """
        if (gdd is None) == (group_delay is None):
            raise MaterialError("give exactly one of gdd and group_delay")
        dispersion = np.asarray(gdd if gdd is not None else group_delay, dtype=float)
        columns = [dispersion] if reflectivity is None else [dispersion, reflectivity]
        if any(np.shape(column) != np.shape(wavelength) for column in columns):
            raise MaterialError(
                "a data column and its wavelength axis differ in length"
            )
        zero_low, zero_high = band[0] - taper_width, band[1] + taper_width
        return cls._from_columns(
            np.asarray(wavelength, dtype=float),
            None if reflectivity is None else np.asarray(reflectivity, dtype=float),
            np.asarray(wavelength, dtype=float),
            dispersion,
            group_delay=gdd is None,
            reference=reference_wavelength,
            fit_order=fit_order,
            fit_window=fit_window,
            band=(zero_low, band[0], band[1], zero_high),
            name=name,
        )

    @classmethod
    def _from_columns(  # noqa: PLR0913 - the columns and every choice of the reduction
        cls,
        reflectivity_wavelength: Float[np.ndarray, " n_r"],
        reflectivity: Float[np.ndarray, " n_r"] | None,
        dispersion_wavelength: Float[np.ndarray, " n_d"],
        dispersion: Float[np.ndarray, " n_d"],
        *,
        group_delay: bool,
        reference: float,
        fit_order: int,
        fit_window: tuple[float, float] | None,
        band: tuple[float, float, float, float],
        name: str,
    ) -> "ChirpedMirror":  # noqa: UP037 - beartype evaluates the annotation eagerly
        """Integrate the dispersion to a phase and assemble the mirror.

        Parameters
        ----------
        reflectivity_wavelength : numpy.ndarray
            Wavelengths of the reflectivity samples, m.
        reflectivity : numpy.ndarray or None
            Power reflectivity in [0, 1], or ``None``.
        dispersion_wavelength : numpy.ndarray
            Wavelengths of the dispersion samples, m.
        dispersion : numpy.ndarray
            GDD in s^2, or group delay in s.
        group_delay : bool
            Which of the two `dispersion` holds.
        reference : float
            Wavelength about which the constant and linear phase are removed, m.
        fit_order : int
            Order of the fitted polynomial.
        fit_window : tuple of float or None
            Angular-frequency window of the fit, rad/s.
        band : tuple of float
            The taper's four wavelengths, m.
        name : str
            A label.

        Returns
        -------
        ChirpedMirror
            One reflection.
        """
        omega, values = _ascending(dispersion_wavelength, dispersion)
        if omega.size <= max(fit_order, 3):
            raise MaterialError(
                f"{omega.size} dispersion samples cannot carry a fit of order "
                f"{fit_order} and a cubic spline"
            )
        # Trapezoid rule on the data's own samples. Its error is h^2 / 12 times the
        # change of the integrand's slope across the band, h the sample spacing in
        # omega: the shipped sets are sampled every 0.04 to 0.3 nm, the Thorlabs
        # group-delay table every 2.7 nm, and in every case that is far below the
        # data's own noise. Group delay needs one integral, GDD two.
        phase = cumulative_trapezoid(values, omega, initial=0)
        if not group_delay:
            phase = cumulative_trapezoid(phase, omega, initial=0)
        detuning = (omega - 2.0 * math.pi * C / reference) * _FEMTOSECOND
        chosen = (
            np.ones(omega.shape, dtype=bool)
            if fit_window is None
            else (omega > fit_window[0]) & (omega < fit_window[1])
        )
        fitted = np.polynomial.Polynomial.fit(
            detuning[chosen], phase[chosen], fit_order
        ).convert()
        # The integration constants are arbitrary: an absolute phase and an overall
        # delay. The fit's higher orders only make its first two robust.
        # `convert` trims trailing zeros, so a flat phase leaves fewer than two.
        constant, slope = np.concatenate([fitted.coef, np.zeros(2)])[:2]
        phase = phase - (constant + slope * detuning)
        if reflectivity is None:
            amplitude_omega = np.empty(0)
            amplitude = np.empty(0)
        else:
            if np.any(reflectivity < 0.0) or np.any(reflectivity > 1.0):
                raise MaterialError("a power reflectivity lies outside [0, 1]")
            amplitude_omega, power = _ascending(reflectivity_wavelength, reflectivity)
            amplitude = np.sqrt(power)
        return cls(omega, phase, amplitude_omega, amplitude, band, 1, name)

    @classmethod
    def lookup(cls, name: str) -> "ChirpedMirror":  # noqa: UP037 - see from_data
        """Return one reflection from a shipped data set.

        Parameters
        ----------
        name : str
            One of :func:`chirped_mirrors`.

        Returns
        -------
        ChirpedMirror
            The mirror, reduced from its data file with the recorded reference
            wavelength, fit order and window, and band.

        Raises
        ------
        MaterialError
            If the name is unknown.

        Examples
        --------
        >>> from solprop.optics import ChirpedMirror
        >>> ChirpedMirror.lookup("HD120").name
        'HD120'
        """
        if name not in _MIRRORS:
            raise MaterialError(
                f"unknown chirped mirror {name!r}; solprop ships {', '.join(_MIRRORS)}"
            )
        record = _MIRRORS[name]
        if record.reflectivity is None:
            reflectivity_wavelength, reflectivity = np.empty(0), None
        else:
            reflectivity_wavelength, percent = _mean_columns(
                _read_columns(record.files[0], record, record.reflectivity),
                geometric=True,
            )
            reflectivity = percent / 100.0
        dispersion_wavelength, dispersion = _mean_columns(
            _read_columns(record.files[-1], record, record.dispersion), geometric=False
        )
        scale = _FEMTOSECOND if record.group_delay else _FEMTOSECOND**2
        return cls._from_columns(
            reflectivity_wavelength,
            reflectivity,
            dispersion_wavelength,
            dispersion * scale,
            group_delay=record.group_delay,
            reference=record.reference,
            fit_order=record.fit_order,
            fit_window=record.fit_window,
            band=record.band,
            name=name,
        )

    def bounces(self, reflections: int) -> "ChirpedMirror":  # noqa: UP037 - see from_data
        """Return the same mirror with another number of reflections.

        Parameters
        ----------
        reflections : int
            Number of reflections; negative for the phase-only inverse.

        Returns
        -------
        ChirpedMirror
            A copy sharing the data.
        """
        return ChirpedMirror(
            self.omega,
            self.phase,
            self.reflectivity_omega,
            self.amplitude,
            self.band,
            int(reflections),
            self.name,
        )

    def gdd(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the group-delay dispersion of one reflection.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s inside the data.

        Returns
        -------
        numpy.ndarray
            The second derivative of the stored phase in s^2; NaN outside the data.
        """
        return np.asarray(self._phase_spline(omega, 2))

    def group_delay(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the group delay of one reflection, less the removed constant.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s inside the data.

        Returns
        -------
        numpy.ndarray
            The first derivative of the stored phase in s; NaN outside the data.
        """
        return np.asarray(self._phase_spline(omega, 1))

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the response of all the reflections.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.

        Returns
        -------
        numpy.ndarray
            ``(sqrt(R) W exp(i phi))**n`` for ``n > 0`` reflections, one for none, and
            ``W exp(i n phi)`` for ``n < 0``; zero outside the band, at and below zero
            frequency, and wherever the data do not reach.
        """
        values = np.asarray(omega, dtype=float)
        if self.reflections == 0:
            return np.ones(values.shape, dtype=np.complex128)
        response = np.zeros(values.shape, dtype=np.complex128)
        positive = values > 0.0
        used = values[positive]
        window = planck_taper(2.0 * math.pi * C / used, *self.band)
        phase = self._phase_spline(used)
        known = np.isfinite(phase)
        count = abs(self.reflections)
        total_phase = self.reflections * np.where(known, phase, 0.0)
        if self.reflections > 0:
            amplitude = window
            if self._amplitude_spline is not None:
                # Splined in wavelength, as data sheets are sampled;
                # clipped defensively, because a cubic through a ragged band edge could
                # overshoot (none of the shipped sets does) and a mirror neither
                # amplifies nor reflects a negative amplitude.
                reflected = self._amplitude_spline(2.0 * math.pi * C / used)
                reflected = np.where(np.isfinite(reflected), reflected, 0.0)
                amplitude = amplitude * np.clip(reflected, 0.0, 1.0)
            strength = amplitude**count
        else:
            # Undoing a mirror's phase: the opposite phase, the taper once, and no
            # amplitude, since a reflectivity cannot be undone.
            strength = window
        response[positive] = np.where(known, strength * np.exp(1j * total_phase), 0.0)
        return response


def _ascending(
    wavelength: Float[np.ndarray, " n_wavelength"],
    values: Float[np.ndarray, " n_values"],
) -> tuple[Float[np.ndarray, " n_wavelength"], Float[np.ndarray, " n_values"]]:
    """Return samples given against wavelength as ascending in angular frequency.

    Parameters
    ----------
    wavelength : numpy.ndarray
        Wavelengths in m, strictly monotonic and positive.
    values : numpy.ndarray
        The samples.

    Returns
    -------
    tuple of numpy.ndarray
        Angular frequency in rad/s, ascending, and the samples in that order.

    Raises
    ------
    MaterialError
        If the wavelengths are not positive and strictly monotonic.
    """
    steps = np.diff(wavelength)
    if np.any(wavelength <= 0.0) or not (np.all(steps > 0.0) or np.all(steps < 0.0)):
        raise MaterialError("the wavelength axis must be positive and monotonic")
    omega = 2.0 * math.pi * C / wavelength
    order = np.argsort(omega)
    return omega[order], values[order]


@dataclass(frozen=True, eq=False)
class Chain(Displayed):
    """Several elements in series.

    Attributes
    ----------
    elements : sequence of TransferFunction
        The elements; their order does not matter, since each is diagonal in frequency.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.optics import Chain, TaylorPhase
    >>> pair = Chain((TaylorPhase((0, 0, 40e-30), 800e-9),
    ...               TaylorPhase((0, 0, -40e-30), 800e-9)))
    >>> bool(np.allclose(pair(np.linspace(2e15, 3e15, 7)), 1.0))
    True
    """

    elements: Sequence[TransferFunction]

    def __repr__(self) -> str:
        """Return a one-line description: how many elements, and the first.

        Returns
        -------
        str
            For example ``Chain(2 elements: ChirpedMirror(...), ...)``.
        """
        count = len(self.elements)
        first = f": {self.elements[0]!r}" if count else ""
        rest = ", ..." if count > 1 else ""
        return f"Chain({count} elements{first}{rest})"

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return f"Chain: {len(self.elements)} elements in series", [
            (str(index), repr(element)) for index, element in enumerate(self.elements)
        ]

    def __post_init__(self) -> None:
        """Freeze the sequence."""
        object.__setattr__(self, "elements", tuple(self.elements))

    def __call__(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the product of the elements' responses.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.

        Returns
        -------
        numpy.ndarray
            ``prod_k H_k(omega)``; one for an empty chain.
        """
        response = np.ones(np.shape(omega), dtype=np.complex128)
        for element in self.elements:
            response = response * element(omega)
        return response


def _pair_frequencies(omega: ArrayLike | float) -> Float[np.ndarray, " n_omega"]:
    """Normalise a real, finite, one-dimensional frequency array without mutation.

    Parameters
    ----------
    omega : array_like
        Angular frequencies in rad/s, shape (n_omega,).

    Returns
    -------
    numpy.ndarray
        Finite real one-dimensional array with unchanged values.
    """
    if np.iscomplexobj(omega):
        raise MaterialError("angular frequencies must be real")
    try:
        values = np.asarray(omega, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MaterialError("angular frequencies must be a real array") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise MaterialError("angular frequencies must be finite and one-dimensional")
    return values


def _pair_reference(wavelength: float, band: tuple[float, float] | None) -> None:
    """Check the reference and the fixed wavelength band.

    Parameters
    ----------
    wavelength : float
        Vacuum reference wavelength in m.
    band : tuple of float or None
        Fixed inclusive ascending wavelength band in m.
    """
    if not math.isfinite(wavelength) or wavelength <= 0.0:
        raise MaterialError("reference wavelength must be finite and positive")
    if band is not None and (
        len(band) != 2
        or not all(math.isfinite(value) for value in band)
        or not 0.0 < band[0] < band[1]
        or not band[0] <= wavelength <= band[1]
    ):
        raise MaterialError("band must be positive, ordered and contain the reference")
    if any(
        not math.isfinite(2.0 * math.pi * C / float(value))
        for value in (wavelength, *(band or ()))
    ):
        raise MaterialError("wavelength gives an unrepresentable angular frequency")


def _finite_pair_phase(
    phase: Float[np.ndarray, " n_omega"],
) -> Float[np.ndarray, " n_omega"]:
    """Reject unrepresentable phases before forming a complex response.

    Parameters
    ----------
    phase : numpy.ndarray
        Computed residual phase in rad, shape (n_omega,).

    Returns
    -------
    numpy.ndarray
        The finite phase with the same shape.

    Raises
    ------
    MaterialError
        If floating-point range was exceeded by the settings or frequencies.
    """
    if not np.all(np.isfinite(phase)):
        raise MaterialError(
            "pair phase is not finite; reduce the numerical range of the settings"
        )
    return phase


def _pair_selected(
    omega: Float[np.ndarray, " n_omega"], band: tuple[float, float] | None
) -> Bool[np.ndarray, " n_omega"]:
    """Select positive frequencies in the fixed inclusive wavelength band.

    Parameters
    ----------
    omega : array_like
        Angular frequencies in rad/s, shape (n_omega,).
    band : tuple of float or None
        Fixed inclusive ascending wavelength band in m.

    Returns
    -------
    numpy.ndarray
        Boolean selection mask, shape (n_omega,).
    """
    selected = omega > 0.0
    if band is not None:
        selected = selected & (omega >= 2.0 * math.pi * C / band[1])
        selected = selected & (omega <= 2.0 * math.pi * C / band[0])
    return selected


def _pair_phase_domain(
    omega: ArrayLike | float, band: tuple[float, float] | None
) -> Float[np.ndarray, " n_omega"]:
    """Require frequencies for which a transmitted phase or ray is requested.

    Parameters
    ----------
    omega : array_like
        Angular frequencies in rad/s, shape (n_omega,).
    band : tuple of float or None
        Fixed inclusive ascending wavelength band in m.

    Returns
    -------
    numpy.ndarray
        Finite positive in-band angular frequencies, shape (n_omega,).
    """
    values = _pair_frequencies(omega)
    if not np.all(_pair_selected(values, band)):
        raise MaterialError("phase and ray paths require positive in-band frequencies")
    return values


def _grating_parameters(line_density: float, order: int) -> None:
    """Validate density in lines per metre and a signed integer diffraction order.

    Parameters
    ----------
    line_density : float
        Positive grating density in lines/m.
    order : int
        Nonzero signed diffraction order.
    """
    if not math.isfinite(line_density) or line_density <= 0.0:
        raise MaterialError("line_density must be finite and positive, in lines/m")
    if isinstance(order, bool) or not isinstance(order, int) or order == 0:
        raise MaterialError("order must be a nonzero signed integer")


def littrow_angle(line_density: float, wavelength: float, order: int = -1) -> float:
    """Return the signed Littrow incidence angle of a reflection grating.

    Parameters
    ----------
    line_density : float
        Grooves per metre, finite and positive; 1200 lines/mm is ``1.2e6``.
    wavelength : float
        Vacuum wavelength in m, finite and positive.
    order : int, optional
        Nonzero signed order in ``sin(theta_m) = sin(theta_i) + m*wavelength/d``.

    Returns
    -------
    float
        Incidence angle in rad, strictly between -pi/2 and pi/2.

    Raises
    ------
    MaterialError
        If density, wavelength or order is invalid.
    OpticsGeometryError
        If the Littrow order is unavailable or grazing.
    """
    _grating_parameters(line_density, order)
    _pair_reference(wavelength, None)
    sine = -order * wavelength * line_density / 2.0
    if not abs(sine) < 1.0:
        raise OpticsGeometryError("Littrow requires a real, non-grazing order")
    return math.asin(sine)


@dataclass(frozen=True)
class GratingPair(Displayed):
    """Apply the dispersion of parallel reflection gratings in air.

    Include the groove phase and remove phase and group delay at `wavelength`.
    The model describes sampled diffraction orders, with no efficiency, aperture,
    polarisation or spatial propagation. See the compression manual for Treacy's
    construction and the meaning of a compensating return pass.

    Attributes
    ----------
    line_density : float
        Grooves per metre, finite and positive.
    separation : float
        Normal distance between the grating surfaces in m, finite and nonnegative.
    incidence_angle : float
        Signed incidence in rad, strictly between -pi/2 and pi/2.
    wavelength : float
        Explicit vacuum reference wavelength in m, finite and positive.
    order : int
        Nonzero signed diffraction order, -1 by default.
    double_pass : bool
        Two traversals (four diffractions) if true; one traversal otherwise.
    band : tuple of float or None
        Fixed ascending vacuum wavelength band in m containing `wavelength`.
        The transfer is zero outside it. Endpoints and evaluated bins are checked;
        no continuous validity certificate is implied.

    Raises
    ------
    MaterialError
        If the parameters or fixed band are invalid.
    OpticsGeometryError
        If an evaluated order is not real or is grazing.
    """

    line_density: float
    separation: float
    incidence_angle: float
    wavelength: float
    order: int = -1
    double_pass: bool = True
    band: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate the settings, reference and band endpoints."""
        _grating_parameters(self.line_density, self.order)
        _pair_reference(self.wavelength, self.band)
        if not math.isfinite(self.separation) or self.separation < 0.0:
            raise MaterialError("separation must be finite and nonnegative")
        if not -math.pi / 2.0 < self.incidence_angle < math.pi / 2.0:
            raise MaterialError(
                "incidence_angle must lie strictly between -pi/2 and pi/2"
            )
        if not isinstance(self.double_pass, bool):
            raise MaterialError("double_pass must be boolean")
        wavelengths = [self.wavelength, *(self.band or ())]
        self._cosine(2.0 * math.pi * C / np.asarray(wavelengths))

    def _cosine(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Select the forward, non-grazing diffraction branch.

        Parameters
        ----------
        omega : array_like
            Angular frequencies in rad/s, shape (n_omega,).

        Returns
        -------
        numpy.ndarray
            Positive diffraction cosine, shape (n_omega,).
        """
        with np.errstate(over="ignore", divide="ignore"):
            sine = math.sin(self.incidence_angle) + (
                2.0 * math.pi * C * self.order * self.line_density / omega
            )
        if np.any(~np.isfinite(sine) | (np.abs(sine) >= 1.0)):
            raise OpticsGeometryError(
                "grating order is unavailable or grazing in the working band"
            )
        return np.sqrt((1.0 - sine) * (1.0 + sine))

    def phase(self, omega: ArrayLike | float) -> Float[np.ndarray, " n_omega"]:
        """Return phase after removing reference phase and group delay.

        Parameters
        ----------
        omega : array_like
            Positive in-band angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        numpy.ndarray
            Optics-convention residual phase in rad, shape ``(n_omega,)``.

        Raises
        ------
        MaterialError
            If frequencies are invalid or the phase exceeds floating-point range.
        OpticsGeometryError
            If any requested diffraction order is unavailable or grazing.
        """
        values = _pair_phase_domain(omega, self.band)
        cosine = self._cosine(values)
        reference = 2.0 * math.pi * C / self.wavelength
        sine = math.sin(self.incidence_angle)
        scaled_order = self.order * self.wavelength * self.line_density
        cosine_ref = float(self._cosine(np.array([reference]))[0])
        detuning = (values - reference) / reference
        slope = (1.0 - sine**2 - sine * scaled_order) / cosine_ref
        # Rationalise F-F0-F0'*(omega-omega0). Subtracting the raw 1e7-rad
        # phases loses their small curvature; the exact remainder starts at delta^2.
        denominator = cosine_ref**2 * (
            values / reference * cosine + cosine_ref + slope * detuning
        )
        passes = 2 if self.double_pass else 1
        with np.errstate(over="ignore", invalid="ignore"):
            phase = np.asarray(
                -passes
                * self.separation
                * reference
                / C
                * scaled_order**2
                * detuning
                * (detuning / denominator)
            )
        return _finite_pair_phase(phase)

    def __call__(self, omega: ArrayLike | float) -> Complex[np.ndarray, " n_omega"]:
        """Return the optics-convention response with fixed band selection.

        Parameters
        ----------
        omega : array_like
            Finite real angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        numpy.ndarray
            ``exp(+i*phase)`` in band, zero outside it and at nonpositive frequency.

        Raises
        ------
        MaterialError
            If the frequency array is invalid.
        OpticsGeometryError
            If any selected diffraction order is unavailable or grazing.
        """
        values = _pair_frequencies(omega)
        selected = _pair_selected(values, self.band)
        response = np.zeros(values.size, dtype=np.complex128)
        response[selected] = np.exp(1j * self.phase(values[selected]))
        return response


def _prism_glass(medium: GlassIndex | str) -> GlassIndex:
    """Resolve only the existing glass registry, never a gas or mixture.

    Parameters
    ----------
    medium : GlassIndex or str
        Glass model or registry name.

    Returns
    -------
    GlassIndex
        Resolved glass model.
    """
    return glass(medium) if isinstance(medium, str) else medium


def _prism_index(
    medium: GlassIndex, omega: Float[np.ndarray, " n_omega"]
) -> Float[np.ndarray, " n_omega"]:
    """Require a finite refractive index above the ambient index of one.

    Parameters
    ----------
    medium : GlassIndex or str
        Glass model or registry name.
    omega : array_like
        Angular frequencies in rad/s, shape (n_omega,).

    Returns
    -------
    numpy.ndarray
        Finite real index above one, shape (n_omega,).
    """
    if omega.size == 0:
        return np.empty(0, dtype=float)
    index = np.real(medium.index(omega))
    if index.shape != omega.shape or np.any(~np.isfinite(index) | (index <= 1.0)):
        raise MaterialError(
            "prism real refractive index must be finite and greater than one"
        )
    return index


def brewster_angle(medium: GlassIndex | str, wavelength: float) -> float:
    """Return the external Brewster incidence angle in air.

    Parameters
    ----------
    medium : GlassIndex or str
        Glass model or existing glass registry name; its real index is used.
    wavelength : float
        Vacuum wavelength in m, finite and positive.

    Returns
    -------
    float
        Incidence angle in rad, ``atan(real(n))``.

    Raises
    ------
    MaterialError
        If the wavelength or glass is invalid, or its real index is not above one.
    """
    _pair_reference(wavelength, None)
    index = _prism_index(
        _prism_glass(medium), np.array([2.0 * math.pi * C / wavelength])
    )
    return math.atan(float(index[0]))


def brewster_apex(medium: GlassIndex | str, wavelength: float) -> float:
    """Return the apex angle for Brewster entry and minimum deviation.

    Parameters
    ----------
    medium : GlassIndex or str
        Glass model or existing glass registry name; its real index is used.
    wavelength : float
        Vacuum wavelength in m, finite and positive.

    Returns
    -------
    float
        Apex angle in rad, ``pi - 2*brewster_angle(medium, wavelength)``.

    Raises
    ------
    MaterialError
        If the reference wavelength or glass index is invalid.
    """
    return math.pi - 2.0 * brewster_angle(medium, wavelength)


@dataclass(frozen=True)
class PrismGeometry(Displayed):
    """Specify fixed first-prism geometry and two second-prism stage coordinates.

    The first apex is (0, 0), with its base horizontal below it. The inverted
    second apex is ``R*u0 + (0, -H)``, where ``u0`` is the reference inter-prism
    ray. Increasing insertion moves only the second prism down; increasing separation
    moves it along ``u0``. Neither coordinate is the physical tip distance.

    Attributes
    ----------
    separation : float
        Stage coordinate R in m from formal zero insertion, finite and nonnegative.
    insertion : float
        Second-prism displacement H perpendicular to its base in m, nonnegative.
    first_face_distance : float
        Distance a along the first entry face from its apex in m, finite and positive.
    apex_angle : float
        Common apex angle in rad, strictly between zero and pi.
    incidence_angle : float
        Fixed external incidence in rad, strictly between zero and pi/2.

    Raises
    ------
    MaterialError
        If a length or angle is outside its domain. Ray feasibility is checked by
        :class:`PrismPair`; zero insertion with positive a does not describe a ray.
    """

    separation: float
    insertion: float
    first_face_distance: float
    apex_angle: float
    incidence_angle: float

    def __post_init__(self) -> None:
        """Check finite stage coordinates, entry distance and angles."""
        for name in ("separation", "insertion", "first_face_distance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise MaterialError(f"{name} must be finite and nonnegative")
        if self.first_face_distance == 0.0:
            raise MaterialError("first_face_distance must be positive")
        if not 0.0 < self.apex_angle < math.pi:
            raise MaterialError("apex_angle must lie strictly between zero and pi")
        if not 0.0 < self.incidence_angle < math.pi / 2.0:
            raise MaterialError(
                "incidence_angle must lie strictly between zero and pi/2"
            )


@dataclass(frozen=True, eq=False)
class PrismPaths(Displayed):
    """Report central-ray paths for one traversal through two ideal wedges.

    Attributes
    ----------
    first_glass : numpy.ndarray
        First glass path in m, shape ``(n_omega,)``.
    air_gap : numpy.ndarray
        Actual inter-prism ray length in m, shape ``(n_omega,)``.
    second_glass : numpy.ndarray
        Second glass path in m, shape ``(n_omega,)``.
    first_exit_face_distance : numpy.ndarray
        First exit distance along its face from the apex in m, shape ``(n_omega,)``.
    second_entry_face_distance : numpy.ndarray
        Second entry distance along its face from the apex in m, shape ``(n_omega,)``.
    second_exit_face_distance : numpy.ndarray
        Second exit distance along its face from the apex in m, shape ``(n_omega,)``.
    """

    first_glass: Float[np.ndarray, " n_omega"]
    air_gap: Float[np.ndarray, " n_omega"]
    second_glass: Float[np.ndarray, " n_omega"]
    first_exit_face_distance: Float[np.ndarray, " n_omega"]
    second_entry_face_distance: Float[np.ndarray, " n_omega"]
    second_exit_face_distance: Float[np.ndarray, " n_omega"]


# Unit vectors make this a dimensionless angular classification tolerance. Length
# tests below scale it by the local coordinates and intersection conditioning; no
# negative path is ever rounded or clamped into a transmitted ray.
_RAY_ROUNDOFF = 64.0 * np.finfo(float).eps


def _prism_refract(
    direction: Float[np.ndarray, "n_omega 2"],
    normal: Float[np.ndarray, " 2"],
    ratio: Float[np.ndarray, " n_omega"],
) -> Float[np.ndarray, "n_omega 2"]:
    """Apply vector Snell on the intended forward face, refusing TIR and grazing.

    Parameters
    ----------
    direction : numpy.ndarray
        Unit ray directions, shape (n_omega, 2).
    normal : numpy.ndarray
        Unit normal directed into the next medium, shape (2,).
    ratio : numpy.ndarray
        Incident over transmitted index, shape (n_omega,).

    Returns
    -------
    numpy.ndarray
        Transmitted unit directions, shape (n_omega, 2).
    """
    cosine = direction @ normal
    tangent = ratio[:, None] * (direction - cosine[:, None] * normal)
    discriminant = 1.0 - np.sum(tangent**2, axis=1)
    if np.any(cosine <= _RAY_ROUNDOFF) or np.any(discriminant <= _RAY_ROUNDOFF**2):
        raise OpticsGeometryError(
            "prism face is backwards, grazing or totally internally reflecting"
        )
    return tangent + np.sqrt(discriminant)[:, None] * normal


def _prism_intersect(
    point: Float[np.ndarray, "n_omega 2"],
    direction: Float[np.ndarray, "n_omega 2"],
    apex: Float[np.ndarray, " 2"],
    face: Float[np.ndarray, " 2"],
) -> tuple[
    Float[np.ndarray, "n_omega 2"],
    Float[np.ndarray, " n_omega"],
    Float[np.ndarray, " n_omega"],
]:
    """Intersect a forward ray with the positive half of the intended prism face.

    Parameters
    ----------
    point : numpy.ndarray
        Current ray positions in m, shape (n_omega, 2).
    direction : numpy.ndarray
        Unit ray directions, shape (n_omega, 2).
    apex : numpy.ndarray
        Target prism apex coordinates in m, shape (2,).
    face : numpy.ndarray
        Unit vector along the target face away from its apex.

    Returns
    -------
    tuple of numpy.ndarray
        Intersection coordinates, forward path lengths and signed face distances, in m.
    """
    offset = apex - point
    determinant = direction[:, 0] * face[1] - direction[:, 1] * face[0]
    if np.any(np.abs(determinant) <= _RAY_ROUNDOFF):
        raise OpticsGeometryError("prism ray grazes a face")
    length = (offset[:, 0] * face[1] - offset[:, 1] * face[0]) / determinant
    distance = (
        offset[:, 0] * direction[:, 1] - offset[:, 1] * direction[:, 0]
    ) / determinant
    scale = np.maximum(np.max(np.abs(point), axis=1), np.max(np.abs(apex)))
    tolerance = _RAY_ROUNDOFF * scale / np.abs(determinant)
    if np.any(~np.isfinite(length) | ~np.isfinite(distance)) or np.any(
        (length <= tolerance) | (distance <= tolerance)
    ):
        raise OpticsGeometryError(
            "prism ray misses the positive face or has a non-forward path"
        )
    return apex + distance[:, None] * face, length, distance


@dataclass(frozen=True, eq=False)
class PrismPair(Displayed):
    """Apply the exact central-ray dispersion of two identical prisms in air.

    Remove phase and analytic group delay at the explicit reference wavelength.
    Glass dispersion is already in the wavefront phase: do not add a separate slab.
    Use only the real index; loss, finite bases, apertures and beam clearance are
    outside this model. A single traversal retains spatial chirp. Doubling assumes
    a compensating return or four-prism arrangement, not an arbitrary return mirror.

    Attributes
    ----------
    medium : GlassIndex or str
        Glass model or name in the existing glass registry; real index must exceed one.
    geometry : PrismGeometry
        Fixed first prism, incidence and apex angle, and second-prism R and H in m.
    wavelength : float
        Explicit vacuum reference wavelength in m, finite and positive.
    double_pass : bool
        Double the pair phase if true, the default.
    band : tuple of float or None
        Fixed ascending vacuum wavelength band in m containing the reference.
        The response is zero outside it. Check endpoints and every evaluated bin;
        this does not prove validity between samples for an arbitrary index model.

    Raises
    ------
    MaterialError
        If a parameter, band or material model is invalid.
    OpticsGeometryError
        If a sampled reference, endpoint or requested ray is infeasible.
    """

    medium: GlassIndex | str
    geometry: PrismGeometry
    wavelength: float
    double_pass: bool = True
    band: tuple[float, float] | None = None
    _glass: GlassIndex = field(init=False, repr=False)
    _exit_reference: float = field(init=False, repr=False)
    _exit_derivative: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Freeze the material and reference ray, then check reference and endpoints."""
        _pair_reference(self.wavelength, self.band)
        if not isinstance(self.double_pass, bool):
            raise MaterialError("double_pass must be boolean")
        medium = _prism_glass(self.medium)
        object.__setattr__(self, "_glass", medium)
        reference = np.array([2.0 * math.pi * C / self.wavelength])
        index = _prism_index(medium, reference)
        angle = self._exit_angle(index)
        derivative = np.real(medium.dindex_domega(reference))
        if derivative.shape != reference.shape or not np.all(np.isfinite(derivative)):
            raise MaterialError(
                "prism index derivative must be finite and match frequencies"
            )
        geometry = self.geometry
        slope = (
            math.sin(geometry.apex_angle)
            * index
            * derivative
            / np.sqrt(index**2 - math.sin(geometry.incidence_angle) ** 2)
            / np.cos(angle)
        )
        object.__setattr__(self, "_exit_reference", float(angle[0]))
        object.__setattr__(self, "_exit_derivative", float(slope[0]))
        self.ray_paths(
            2.0 * math.pi * C / np.asarray([self.wavelength, *(self.band or ())])
        )

    def _exit_angle(
        self, index: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the signed external exit angle of the first prism.

        Parameters
        ----------
        index : numpy.ndarray
            Real refractive index at each frequency, shape (n_omega,).

        Returns
        -------
        numpy.ndarray
            First-prism exit angles in rad, shape (n_omega,).
        """
        geometry = self.geometry
        sine = math.sin(geometry.incidence_angle)
        exit_sine = (
            math.sin(geometry.apex_angle) * np.sqrt(index**2 - sine**2)
            - math.cos(geometry.apex_angle) * sine
        )
        if np.any(np.abs(exit_sine) >= 1.0):
            raise OpticsGeometryError(
                "first prism exit is totally internally reflecting or grazing"
            )
        return np.arcsin(exit_sine)

    def _apex(self) -> Float[np.ndarray, " 2"]:
        """Return the second apex in the fixed laboratory coordinates.

        Returns
        -------
        numpy.ndarray
            Second apex coordinates in m, shape (2,).
        """
        angle = self.geometry.apex_angle / 2.0 - self._exit_reference
        return self.geometry.separation * np.array(
            [math.cos(angle), math.sin(angle)]
        ) - np.array([0.0, self.geometry.insertion])

    @property
    def apex_distance(self) -> float:
        """Return the physical tip-to-tip distance in metres.

        Returns
        -------
        float
            ``norm(R*u0 + (0, -H))``, which changes when either stage moves.
        """
        return float(np.linalg.norm(self._apex()))

    def ray_paths(self, omega: ArrayLike | float) -> PrismPaths:
        """Trace positive central-ray paths through the two ideal wedges.

        Parameters
        ----------
        omega : array_like
            Positive in-band angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        PrismPaths
            New path and face-distance arrays in m, shape ``(n_omega,)``.

        Raises
        ------
        MaterialError
            If frequencies or the real refractive index are invalid.
        OpticsGeometryError
            If any ray is backwards, grazing, misses a face or undergoes TIR.
        """
        values = _pair_phase_domain(omega, self.band)
        index = _prism_index(self._glass, values)
        geometry = self.geometry
        half = geometry.apex_angle / 2.0
        sine, cosine = math.sin(half), math.cos(half)
        normal_in, normal_out = np.array([cosine, -sine]), np.array([cosine, sine])
        incident = geometry.incidence_angle - half
        direction = np.tile([math.cos(incident), math.sin(incident)], (values.size, 1))
        entry = np.tile(
            geometry.first_face_distance * np.array([-sine, -cosine]), (values.size, 1)
        )
        inside = _prism_refract(direction, normal_in, 1.0 / index)
        exit_first, first_glass, first_face = _prism_intersect(
            entry, inside, np.zeros(2), np.array([sine, -cosine])
        )
        between = _prism_refract(inside, normal_out, index)
        second = self._apex()
        entry_second, air, entry_face = _prism_intersect(
            exit_first, between, second, np.array([-sine, cosine])
        )
        inside_second = _prism_refract(between, normal_out, 1.0 / index)
        _, second_glass, exit_face = _prism_intersect(
            entry_second, inside_second, second, np.array([sine, cosine])
        )
        _prism_refract(inside_second, normal_in, index)
        return PrismPaths(
            first_glass, air, second_glass, first_face, entry_face, exit_face
        )

    def phase(self, omega: ArrayLike | float) -> Float[np.ndarray, " n_omega"]:
        """Return wavefront phase with reference phase and group delay removed.

        Parameters
        ----------
        omega : array_like
            Positive in-band angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        numpy.ndarray
            Optics-convention residual phase in rad, shape ``(n_omega,)``.

        Raises
        ------
        MaterialError
            If frequencies or the material model are invalid, or the phase
            is not finite.
        OpticsGeometryError
            If any requested central ray is infeasible.
        """
        values = _pair_phase_domain(omega, self.band)
        self.ray_paths(values)
        index = _prism_index(self._glass, values)
        delta = self._exit_angle(index) - self._exit_reference
        reference = 2.0 * math.pi * C / self.wavelength
        angle = self.geometry.apex_angle / 2.0 - self._exit_reference
        apex = self._apex()
        parallel = float(np.array([math.cos(angle), math.sin(angle)]) @ apex)
        transverse = float(np.array([math.sin(angle), -math.cos(angle)]) @ apex)
        # Wavefront path is u(omega).A2, up to a frequency-independent path.
        # cos(delta)-1 = -2 sin(delta/2)^2 avoids subtracting metre-scale paths.
        with np.errstate(over="ignore", invalid="ignore"):
            residual = values * (
                -2.0 * parallel * np.sin(delta / 2.0) ** 2 + transverse * np.sin(delta)
            ) - reference * transverse * self._exit_derivative * (values - reference)
            phase = np.asarray((2 if self.double_pass else 1) / C * residual)
        return _finite_pair_phase(phase)

    def __call__(self, omega: ArrayLike | float) -> Complex[np.ndarray, " n_omega"]:
        """Return the phase response with fixed band selection.

        Parameters
        ----------
        omega : array_like
            Finite real angular frequencies in rad/s, shape ``(n_omega,)``.

        Returns
        -------
        numpy.ndarray
            ``exp(+i*phase)`` in band, zero outside it and at nonpositive frequency.

        Raises
        ------
        MaterialError
            If frequencies or the material model are invalid, or the phase
            is not finite.
        OpticsGeometryError
            If any selected central ray is infeasible.
        """
        values = _pair_frequencies(omega)
        selected = _pair_selected(values, self.band)
        response = np.zeros(values.size, dtype=np.complex128)
        response[selected] = np.exp(1j * self.phase(values[selected]))
        return response

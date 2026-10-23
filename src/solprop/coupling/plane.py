# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""One physical plane of an ultrafast field, with everything needed to use it.

A :class:`FieldPlane` binds a field's **native coefficients** to what they mean
(SEP 31, sections 4.1, 4.5 and 5.1): the representation and its units, the temporal
lattice, the reference the spectrum is written in, the oriented plane, the medium and
power metric there, the scoped path certificate of the run that produced it, and its
provenance. It is immutable and owns read-only copies of its arrays, so a validated
plane cannot be changed through an alias.

**Samples.** The stored array is what an output file stores: the optics-convention
raw forward DFT of each column, referenced to the centre of the time window
(:func:`solprop.fourier.spectrum_to_optics`), shape ``(n_modes, n_pol, n_omega)``.
Its columns are

=================  ==================================================================
``kind``           columns and units
=================  ==================================================================
``modal``          true-power coefficients ``b = sqrt(Re n_eff) a`` of an ordinary
                   mode set, sqrt(W); metric the identity
``general_modal``  general-modal coordinates, sqrt(W); metric the basis Gram
``radial``         order-zero Hankel spectrum of the electric field, V m
``cartesian``      transverse raw DFT of the electric field, V/m
``planar``         slab amplitude per y frequency, sqrt(W/m)
=================  ==================================================================

**Reference.** A run's spectrum is in its moving frame. The absolute
optics-convention spectrum is the stored one times
:math:`\exp[i(\phi + \omega\tau)]` (:class:`PlaneReference`), with ω the absolute
angular frequency of each bin: τ is the absolute time of the record's ``t = 0`` and
φ a carrier phase. The pair is the frame integral the linear operator removed,
:math:`\int R\,dz`, so a long path never has to translate a femtosecond pulse
across its window.

The plane is a P-layer value type: NumPy and other physics modules only. Extracting
one from a run is :func:`solprop.processing.field_plane`, and a receiving simulation
is described by :class:`solprop.coupling.InputPort`.
"""

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop import fourier
from solprop.exceptions import CouplingError
from solprop.grid import FreeGrid, RadialGrid, TimeGrid
from solprop.modes.protocol import Mode
from solprop.modes.slab import PlanarGrid
from solprop.spatial import FluxSide, SpatialFlux, SpatialMedium

__all__ = [
    "PLANE_KINDS",
    "Calibration",
    "Exterior",
    "FieldPlane",
    "NodalField",
    "PlaneBasis",
    "PlaneGeometry",
    "PlaneKind",
    "PlaneMetric",
    "PlaneProvenance",
    "PlaneReference",
    "basis_identity",
    "lattice_identity",
    "modal_basis_identity",
    "reference_at",
    "space_record",
]

PlaneKind = Literal["modal", "general_modal", "radial", "cartesian", "planar"]
"""The native representation of a plane's columns."""

PLANE_KINDS: tuple[PlaneKind, ...] = (
    "modal",
    "general_modal",
    "radial",
    "cartesian",
    "planar",
)
"""Every :data:`PlaneKind`, in a fixed order."""

Exterior = Literal["zero", "unknown"]
"""What a finite record says outside its samples: zero, or not known."""

Calibration = Literal["phase", "delay"]
"""An affine reference term a plane knows absolutely."""

MetricKind = Literal["identity", "dense", "admittance", "unknown"]
"""How a plane's native power metric is represented."""

_UNIT_TOLERANCE = 1e-12
"""Allowed departure of a declared axis from unit length and orthogonality."""


def _frozen(
    values: object, dtype: type, name: str, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """Return an owned, read-only, finite copy of an array.

    Parameters
    ----------
    values : object
        Anything :func:`numpy.array` accepts.
    dtype : type
        The dtype of the copy.
    name : str
        The attribute name, for the refusal.
    shape : tuple of int or None, optional
        The required shape, when one is fixed.

    Returns
    -------
    numpy.ndarray
        The copy, with ``flags.writeable`` False.

    Raises
    ------
    CouplingError
        If the array is not finite or has the wrong shape.
    """
    array = np.array(values, dtype=dtype, copy=True)
    if shape is not None and array.shape != shape:
        msg = f"{name} has shape {array.shape}, expected {shape}"
        raise CouplingError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite"
        raise CouplingError(msg)
    array.setflags(write=False)
    return array


def _canonical(record: object) -> str:
    """Return the canonical JSON text an identity is a digest of.

    Parameters
    ----------
    record : object
        JSON-ready data.

    Returns
    -------
    str
        Sorted, compact JSON. A material's open interval bound is infinite, and
        JSON's ``Infinity`` spells it deterministically.
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def basis_identity(route: PlaneKind, record: Mapping[str, object]) -> str:
    """Return the identity of a plane's basis: a digest of its route and record.

    The one function both ends of a transfer call, so that a plane read from a
    file and the port of a receiving simulation compare the same digest. The
    record is JSON-ready: for a spatial route, :func:`space_record` of the grid; for
    a modal route, the run's serialised mode records and general-modal identity.

    Parameters
    ----------
    route : {'modal', 'general_modal', 'radial', 'cartesian', 'planar'}
        The representation.
    record : collections.abc.Mapping
        The JSON-ready description of the basis.

    Returns
    -------
    str
        Sixty-four lowercase hexadecimal characters.

    Examples
    --------
    >>> from solprop.coupling import basis_identity, space_record
    >>> from solprop.grid import RadialGrid
    >>> record = space_record(RadialGrid(1e-3, 64))
    >>> basis_identity("radial", record) == basis_identity("radial", dict(record))
    True
    """
    text = _canonical({"route": route, "record": dict(record)})
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_QUANTUM_DIGITS = 12
"""Significant digits a modal basis's sampled quantities enter its identity with.

The same guide evaluated at the same plane agrees to a few ulp whatever path led
there (a constant fill of another length, a gradient met at the same pressure); twelve
digits sit well above that and far below any physical difference of guide."""


def _quantised(values: np.ndarray) -> list[float]:
    """Return values rounded to :data:`_QUANTUM_DIGITS` significant digits.

    Parameters
    ----------
    values : numpy.ndarray
        Finite real values.

    Returns
    -------
    list of float
        The rounded values.
    """
    return [float(f"{float(one):.{_QUANTUM_DIGITS}g}") for one in np.ravel(values)]


def modal_basis_identity(modes: Sequence[Mode], grid: TimeGrid, z: float) -> str:
    """Return the identity of an ordinary mode set's basis at one plane.

    An ordinary mode's saved coefficient is a true-power amplitude of the mode's
    own profile at the plane, so the basis there is fixed by each mode's class and
    channel label, its effective area and its real effective index on the grid's
    active band at that plane, quantised to twelve significant digits. The guide's
    path elsewhere does not enter: the exit of one capillary and the entrance of an
    identical one of another length are the same basis.

    Parameters
    ----------
    modes : sequence of solprop.modes.protocol.Mode
        The mode set, in state order.
    grid : solprop.grid.TimeGrid
        The temporal grid; its active band is sampled.
    z : float
        The plane's position along the guide, m.

    Returns
    -------
    str
        Sixty-four lowercase hexadecimal characters.
    """
    omega = grid.omega[grid.active_mask & (grid.omega > 0)]
    record = {
        "modes": [
            {
                "kind": f"{type(mode).__module__}.{type(mode).__qualname__}",
                "label": dict(mode.channel_label.as_mapping()),
                "aeff": _quantised(np.array([mode.aeff(z)])),
                "neff": _quantised(np.real(np.asarray(mode.neff(omega, z)))),
            }
            for mode in modes
        ],
        "lattice": lattice_identity(grid),
    }
    return basis_identity("modal", record)


def space_record(space: RadialGrid | FreeGrid | PlanarGrid) -> dict[str, object]:
    """Return the JSON-ready record of a transverse grid: its class and arguments.

    The grid's constructor arguments determine every axis, weight and transform, so
    they are its identity; the record has the form the output layer serialises a
    grid in, which is what makes a plane read from a file comparable with a live
    grid. A :class:`~solprop.modes.slab.PlanarGrid` carries slab modes and has no
    such record.

    Parameters
    ----------
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The transverse grid.

    Returns
    -------
    dict
        ``{"__class__": "module.Name", <init fields>...}``.

    Raises
    ------
    CouplingError
        For a planar grid.

    Examples
    --------
    >>> from solprop.coupling import space_record
    >>> from solprop.grid import RadialGrid
    >>> space_record(RadialGrid(1e-3, 64))["n_r"]
    64
    """
    if isinstance(space, PlanarGrid):
        msg = (
            "a planar grid's columns are slab amplitudes of its own modes and it has "
            "no plain grid record; planar planes are not exchanged in this release"
        )
        raise CouplingError(msg)
    kind = type(space)
    record: dict[str, object] = {"__class__": f"{kind.__module__}.{kind.__qualname__}"}
    for one in dataclasses.fields(space):
        if one.init:
            value = getattr(space, one.name)
            record[one.name] = float(value) if isinstance(value, float) else value
    return record


def lattice_identity(grid: TimeGrid) -> str:
    """Return the identity of a temporal lattice: its layout, samples and reference.

    Two grids with the same identity have the same time samples and the same
    absolute frequency of every bin, so their spectra can be copied bin for bin.
    The spectral collar (the requested band) is not part of the lattice.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid.

    Returns
    -------
    str
        Sixty-four lowercase hexadecimal characters.

    Examples
    --------
    >>> from solprop.coupling import lattice_identity
    >>> from solprop.grid import EnvelopeGrid
    >>> one = EnvelopeGrid.from_limits(1.0, 800e-9, (600e-9, 1.2e-6), 1e-12)
    >>> two = EnvelopeGrid.from_limits(2.0, 800e-9, (600e-9, 1.2e-6), 1e-12)
    >>> lattice_identity(one) == lattice_identity(two)
    True
    """
    record = {
        "kind": grid.kind,
        "n_t": grid.n_t,
        "n_to": grid.n_to,
        "dt": float(grid.dt),
        "dt_fine": float(grid.dt_fine),
        "reference_wavelength": float(grid.reference_wavelength),
    }
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PlaneGeometry:
    """An oriented physical plane: its position, side, origin and local axes.

    The local transverse basis is right-handed: ``x_axis`` cross ``y_axis`` =
    ``normal``, and the field's propagation is forward along ``normal``
    (SEP 31, section 4.1). A reflection changes these axes and the polarisation
    together; it is never an unexplained conjugation.

    Attributes
    ----------
    z : float
        Position of the plane along its own path, m.
    side : {'before', 'after'} or None, optional
        Side of a declared index jump at ``z``; None away from one.
    origin : tuple of float, optional
        Physical origin of the transverse coordinates, m, in the laboratory frame.
    normal : tuple of float, optional
        Unit forward normal, laboratory frame. Default ``(0, 0, 1)``.
    x_axis : tuple of float, optional
        Unit local x axis, orthogonal to the normal. Default ``(1, 0, 0)``.

    Raises
    ------
    CouplingError
        If a value is not finite, an axis is not of unit length, or the two axes
        are not orthogonal.

    Examples
    --------
    >>> from solprop.coupling import PlaneGeometry
    >>> PlaneGeometry(0.02).y_axis
    (0.0, 1.0, 0.0)
    """

    z: float
    side: FluxSide | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    x_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Check the position, the side and the axes."""
        values = [self.z, *self.origin, *self.normal, *self.x_axis]
        if not all(math.isfinite(value) for value in values):
            msg = "a plane's position, origin and axes must be finite"
            raise CouplingError(msg)
        normal = np.asarray(self.normal, dtype=float)
        x_axis = np.asarray(self.x_axis, dtype=float)
        if (
            abs(float(np.linalg.norm(normal)) - 1.0) > _UNIT_TOLERANCE
            or abs(float(np.linalg.norm(x_axis)) - 1.0) > _UNIT_TOLERANCE
            or abs(float(normal @ x_axis)) > _UNIT_TOLERANCE
        ):
            msg = (
                "a plane's normal and x axis are orthogonal unit vectors; got normal "
                f"{self.normal} and x axis {self.x_axis}"
            )
            raise CouplingError(msg)
        object.__setattr__(self, "z", float(self.z))
        object.__setattr__(self, "origin", tuple(float(one) for one in self.origin))
        object.__setattr__(self, "normal", tuple(float(one) for one in self.normal))
        object.__setattr__(self, "x_axis", tuple(float(one) for one in self.x_axis))

    @property
    def y_axis(self) -> tuple[float, float, float]:
        """Return the local y axis, ``normal`` cross ``x_axis``.

        Returns
        -------
        tuple of float
            The unit vector completing the right-handed frame.
        """
        value = np.cross(np.asarray(self.normal), np.asarray(self.x_axis))
        return (float(value[0]), float(value[1]), float(value[2]))

    def oriented_as(self, other: "PlaneGeometry") -> bool:  # noqa: UP037 - beartype evaluates the class body
        """Return whether two planes share their normal, axes and origin.

        Parameters
        ----------
        other : PlaneGeometry
            The other plane.

        Returns
        -------
        bool
            True when origin, normal and x axis agree to rounding.
        """
        return all(
            np.allclose(getattr(self, name), getattr(other, name), rtol=0.0, atol=1e-12)
            for name in ("origin", "normal", "x_axis")
        )


@dataclass(frozen=True)
class PlaneReference:
    r"""The affine reference a stored spectrum is written in.

    The absolute optics-convention spectrum is the stored one times
    :math:`\exp[i(\phi + \omega\tau)]`, with ω the absolute angular frequency of
    each bin (SEP 31, section 4.5). In the time domain τ is the absolute time of the
    record's ``t = 0``: the field at record time t happened at ``t + delay``.

    Attributes
    ----------
    phase : float, optional
        Carrier phase φ, rad.
    delay : float, optional
        Time origin τ, s.
    calibrated : tuple of {'phase', 'delay'}, optional
        Which terms are known absolutely. A run knows both from its frame; a
        phase-only optical model may leave the constant unknown.

    Raises
    ------
    CouplingError
        If a value is not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.coupling import PlaneReference
    >>> reference = PlaneReference(phase=0.5, delay=1e-12)
    >>> bool(np.isclose(np.angle(reference.factor(np.array([0.0]))[0]), 0.5))
    True
    """

    phase: float = 0.0
    delay: float = 0.0
    calibrated: tuple[Calibration, ...] = ("phase", "delay")

    def __post_init__(self) -> None:
        """Check the values and order the calibration names."""
        if not (math.isfinite(self.phase) and math.isfinite(self.delay)):
            msg = "a plane reference's phase and delay must be finite"
            raise CouplingError(msg)
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(self, "delay", float(self.delay))
        object.__setattr__(
            self,
            "calibrated",
            tuple(name for name in ("phase", "delay") if name in self.calibrated),
        )

    def factor(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> Complex[np.ndarray, " n_omega"]:
        r"""Return the factor taking stored to absolute spectra.

        Parameters
        ----------
        omega : numpy.ndarray
            Absolute angular frequencies, rad/s, shape ``(n_omega,)``.

        Returns
        -------
        numpy.ndarray
            :math:`\exp[i(\phi + \omega\tau)]`, complex128, shape ``(n_omega,)``.
        """
        return np.exp(1j * (self.phase + np.asarray(omega, dtype=float) * self.delay))

    def minus(self, other: "PlaneReference") -> "PlaneReference":  # noqa: UP037 - beartype evaluates the class body
        """Return this reference measured from another.

        Parameters
        ----------
        other : PlaneReference
            The reference subtracted.

        Returns
        -------
        PlaneReference
            ``(phase - other.phase, delay - other.delay)``, calibrated where both
            are.
        """
        return PlaneReference(
            self.phase - other.phase,
            self.delay - other.delay,
            tuple(name for name in self.calibrated if name in other.calibrated),
        )

    def plus(self, other: "PlaneReference") -> "PlaneReference":  # noqa: UP037 - beartype evaluates the class body
        """Return the sum of two references, calibrated where both are.

        Parameters
        ----------
        other : PlaneReference
            The reference added.

        Returns
        -------
        PlaneReference
            ``(phase + other.phase, delay + other.delay)``.
        """
        return PlaneReference(
            self.phase + other.phase,
            self.delay + other.delay,
            tuple(name for name in self.calibrated if name in other.calibrated),
        )

    def record(self) -> dict[str, object]:
        """Return the JSON-ready record of the reference.

        Returns
        -------
        dict
            ``phase``, ``delay`` and ``calibrated``.
        """
        return {
            "phase": self.phase,
            "delay": self.delay,
            "calibrated": list(self.calibrated),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "PlaneReference":  # noqa: UP037 - beartype evaluates the class body
        """Return the reference a record describes.

        Parameters
        ----------
        record : collections.abc.Mapping
            As :meth:`record` writes it.

        Returns
        -------
        PlaneReference
            The reference.
        """
        return cls(
            float(cast("float", record["phase"])),
            float(cast("float", record["delay"])),
            tuple(cast("Sequence[Calibration]", record["calibrated"])),
        )


def _chebyshev_at(table: Mapping[str, object], z: float) -> float:
    """Evaluate a persisted piecewise Chebyshev series at one position.

    Parameters
    ----------
    table : collections.abc.Mapping
        ``breakpoints`` (increasing interval edges, m) and ``coefficients`` (one
        list per interval, lowest degree first).
    z : float
        The position, inside the span to rounding.

    Returns
    -------
    float
        The value.

    Raises
    ------
    CouplingError
        If ``z`` lies outside the span.
    """
    edges = np.asarray(table["breakpoints"], dtype=float)
    slack = 1e-9 * max(1.0, abs(float(edges[-1])))
    if not float(edges[0]) - slack <= z <= float(edges[-1]) + slack:
        msg = (
            f"z = {z!r} m is outside the frame record's span "
            f"[{edges[0]}, {edges[-1]}] m"
        )
        raise CouplingError(msg)
    coefficients = cast("Sequence[Sequence[float]]", table["coefficients"])
    index = int(np.clip(np.searchsorted(edges, z, side="right") - 1, 0, edges.size - 2))
    lo, hi = float(edges[index]), float(edges[index + 1])
    x = min(1.0, max(-1.0, (2.0 * z - lo - hi) / (hi - lo)))
    return float(np.polynomial.chebyshev.chebval(x, np.asarray(coefficients[index])))


def reference_at(record: Mapping[str, object], z: float) -> PlaneReference:
    r"""Return the reference of a run's plane at ``z`` from its persisted record.

    The record is the run's plane-reference record of output schema 24:
    ``origin``, the reference of the run's own path origin (its parent's, for a run
    launched from another), and ``frame``, the antiderivatives
    :math:`\Phi_0(z)=\int_0^z\beta_0` and :math:`T(z)=\int_0^z\beta_1` of the terms
    its linear operator removed, with the frequency origin :math:`\omega_o` of the
    linear term (the reference frequency on an envelope grid, zero on a field
    grid). Then :math:`\phi = \phi_0 + \Phi_0 - \omega_oT` and
    :math:`\tau = \tau_0 + T`. A record without a frame (an operator supplied
    directly) leaves both terms uncalibrated.

    Parameters
    ----------
    record : collections.abc.Mapping
        The persisted plane-reference record.
    z : float
        Position of the plane along the run's path, m.

    Returns
    -------
    PlaneReference
        The reference of the stored spectrum at ``z``.

    Raises
    ------
    CouplingError
        If ``z`` lies outside the frame record's span.

    Examples
    --------
    >>> from solprop.coupling import reference_at
    >>> record = {
    ...     "origin": {"phase": 0.0, "delay": 1e-12, "calibrated": ["phase", "delay"]},
    ...     "frame": {"frequency_origin": 0.0,
    ...               "phase": {"breakpoints": [0.0, 1.0], "coefficients": [[0.0]]},
    ...               "delay": {"breakpoints": [0.0, 1.0],
    ...                         "coefficients": [[2e-9, 2e-9]]}},
    ... }
    >>> f"{reference_at(record, 1.0).delay:.3e}"
    '4.001e-09'
    """
    origin = PlaneReference.from_record(cast("Mapping[str, object]", record["origin"]))
    frame = record.get("frame")
    if frame is None:
        return PlaneReference(origin.phase, origin.delay, ())
    frame = cast("Mapping[str, object]", frame)
    phase = _chebyshev_at(cast("Mapping[str, object]", frame["phase"]), z)
    delay = _chebyshev_at(cast("Mapping[str, object]", frame["delay"]), z)
    removed = PlaneReference(
        phase - float(cast("float", frame["frequency_origin"])) * delay, delay
    )
    return origin.plus(removed)


@dataclass(frozen=True, eq=False)
class PlaneMetric:
    """The native power metric of a plane's columns at its own position.

    Each kind gives the energy crossing the plane per stored bin as a quadratic
    form of the samples (SEP 31, section 4.3, the metric Q):

    ==============  ==============================================================
    ``identity``    ``e_k sum |s|^2``: ordinary true-power coefficients
    ``dense``       ``e_k s_k^H Q_k s_k``: a general basis Gram, optics convention
    ``admittance``  ``scale e_k sum_j v_j Re(Y_jk) |s_jk|^2``: a spatial plane's
                    axial admittance ``Y = c k_z / omega`` (SEP 34)
    ``unknown``     no energy can be formed; `reason` says why
    ==============  ==============================================================

    with ``e_k`` the bin weights of :func:`solprop.fourier.energy_weights`. A
    dense metric's per-channel diagonal is not a set of channel powers.

    Attributes
    ----------
    kind : {'identity', 'dense', 'admittance', 'unknown'}
        The representation.
    matrix : numpy.ndarray or None, optional
        The optics-convention Gram ``Q = conj(W)``, shape ``(n_modes, n_modes,
        n_omega)``, dimensionless; exactly for ``dense``.
    admittance : numpy.ndarray or None, optional
        Dimensionless ``c k_z / omega`` of each column, shape ``(n_modes,
        n_omega)``; exactly for ``admittance``.
    weights : numpy.ndarray or None, optional
        Transverse Parseval weights of the columns, shape ``(n_modes,)``; with
        ``admittance``.
    scale : float, optional
        Factor turning the weighted sum into joules: ``eps0 c / 2`` for a field in
        V/m, one otherwise.
    flux : solprop.spatial.SpatialFlux or None, optional
        The plane's measure, when its medium is at hand; it can be resolved again
        on another temporal grid.
    reason : str, optional
        Why an ``unknown`` metric is unknown.

    Raises
    ------
    CouplingError
        If the arrays present are not the ones `kind` names, or an unknown metric
        gives no reason.
    """

    kind: MetricKind
    matrix: Complex[np.ndarray, "n_modes n_modes_q n_omega"] | None = None
    admittance: Complex[np.ndarray, "n_modes n_omega"] | None = None
    weights: Float[np.ndarray, " n_modes"] | None = None
    scale: float = 1.0
    flux: SpatialFlux | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        """Own the arrays and check they are the ones the kind names."""
        wanted = {
            "identity": (),
            "dense": ("matrix",),
            "admittance": ("admittance", "weights"),
            "unknown": (),
        }
        present = tuple(
            name
            for name in ("matrix", "admittance", "weights")
            if getattr(self, name) is not None
        )
        if present != wanted[self.kind]:
            msg = f"a {self.kind} metric carries {wanted[self.kind]}, got {present}"
            raise CouplingError(msg)
        if self.kind == "unknown" and not self.reason:
            msg = "an unknown metric says why it is unknown"
            raise CouplingError(msg)
        for name, dtype in (
            ("matrix", np.complex128),
            ("admittance", np.complex128),
            ("weights", np.float64),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _frozen(value, dtype, name))

    @classmethod
    def from_flux(cls, flux: SpatialFlux) -> "PlaneMetric":  # noqa: UP037 - beartype evaluates the class body
        """Return the admittance metric of a resolved spatial measure.

        Parameters
        ----------
        flux : solprop.spatial.SpatialFlux
            The plane's measure (SEP 34).

        Returns
        -------
        PlaneMetric
            The ``admittance`` metric, keeping `flux`.
        """
        return cls(
            "admittance",
            admittance=flux.admittance(),
            weights=flux.space.spectral_weights,
            scale=flux.energy_scale,
            flux=flux,
        )

    def spectral_energy(
        self, samples: Complex[np.ndarray, "n_modes n_pol n_omega"], grid: TimeGrid
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the energy each bin carries through the plane, J.

        Parameters
        ----------
        samples : numpy.ndarray
            Stored spectrum, shape ``(n_modes, n_pol, n_omega)``.
        grid : solprop.grid.TimeGrid
            Its temporal grid.

        Returns
        -------
        numpy.ndarray
            Energy per bin, J (J/m per unit width on a line), shape ``(n_omega,)``.

        Raises
        ------
        CouplingError
            For an ``unknown`` metric, or arrays of the wrong shape.
        """
        bins = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
        if self.kind == "unknown":
            msg = f"this plane's power metric is unknown: {self.reason}"
            raise CouplingError(msg)
        if self.kind == "identity":
            return bins * np.sum(np.abs(samples) ** 2, axis=(0, 1))
        if self.kind == "dense":
            matrix = cast("np.ndarray", self.matrix)
            if matrix.shape != (samples.shape[0], samples.shape[0], samples.shape[2]):
                msg = f"metric shape {matrix.shape} does not match {samples.shape}"
                raise CouplingError(msg)
            form = np.einsum("mpw,mnw,npw->w", np.conj(samples), matrix, samples)
            return bins * np.real(form)
        admittance = cast("np.ndarray", self.admittance)
        weights = cast("np.ndarray", self.weights)
        if admittance.shape != (samples.shape[0], samples.shape[2]):
            msg = f"admittance shape {admittance.shape} does not match {samples.shape}"
            raise CouplingError(msg)
        density = np.sum(np.abs(samples) ** 2, axis=1)
        return (
            self.scale
            * bins
            * np.sum(weights[:, None] * np.real(admittance) * density, axis=0)
        )


@dataclass(frozen=True, eq=False)
class PlaneBasis:
    """What the columns of a plane are, and the identity they are checked by.

    Attributes
    ----------
    kind : {'modal', 'general_modal', 'radial', 'cartesian', 'planar'}
        The representation.
    identity : str
        :func:`basis_identity` of the basis: a plane and a port with the same
        identity have the same columns.
    labels : tuple of str, optional
        Channel labels of a modal basis, in column order.
    space : solprop.grid.RadialGrid, solprop.grid.FreeGrid or None, optional
        The transverse grid of a spatial basis.

    Raises
    ------
    CouplingError
        If the identity is not a digest, or a spatial basis has no grid.
    """

    kind: PlaneKind
    identity: str
    labels: tuple[str, ...] = ()
    space: RadialGrid | FreeGrid | None = None

    def __post_init__(self) -> None:
        """Check the digest and the grid."""
        if len(self.identity) != 64:
            msg = "a basis identity is a SHA-256 hex digest"
            raise CouplingError(msg)
        if self.kind in ("radial", "cartesian") and self.space is None:
            msg = f"a {self.kind} basis carries its transverse grid"
            raise CouplingError(msg)
        object.__setattr__(self, "labels", tuple(self.labels))

    @classmethod
    def of_space(cls, space: RadialGrid | FreeGrid) -> "PlaneBasis":  # noqa: UP037 - beartype evaluates the class body
        """Return the basis of a spatial grid.

        Parameters
        ----------
        space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
            The grid.

        Returns
        -------
        PlaneBasis
            ``radial`` for a radial grid and ``cartesian`` for a Cartesian one.
        """
        kind: PlaneKind = "radial" if isinstance(space, RadialGrid) else "cartesian"
        return cls(kind, basis_identity(kind, space_record(space)), space=space)


@dataclass(frozen=True, eq=False)
class NodalField:
    """A plane's physical electric and magnetic fields at transverse nodes.

    Attributes
    ----------
    nodes : numpy.ndarray
        Node positions ``(x, y)`` in m, shape ``(n_nodes, 2)``.
    weights : numpy.ndarray
        Area weights in m^2, shape ``(n_nodes,)``.
    electric : numpy.ndarray
        Optics-convention raw spectra of ``E_x, E_y, E_z`` in V/m, shape
        ``(n_nodes, 3, n_omega)``.
    magnetic : numpy.ndarray or None, optional
        The same of ``H`` in A/m, when the basis supplies it.
    """

    nodes: Float[np.ndarray, "n_nodes 2"]
    weights: Float[np.ndarray, " n_nodes"]
    electric: Complex[np.ndarray, "n_nodes 3 n_omega"]
    magnetic: Complex[np.ndarray, "n_nodes 3 n_omega"] | None = None

    def __post_init__(self) -> None:
        """Own read-only copies."""
        object.__setattr__(self, "nodes", _frozen(self.nodes, np.float64, "nodes"))
        object.__setattr__(
            self, "weights", _frozen(self.weights, np.float64, "weights")
        )
        object.__setattr__(
            self, "electric", _frozen(self.electric, np.complex128, "electric")
        )
        if self.magnetic is not None:
            object.__setattr__(
                self,
                "magnetic",
                _frozen(self.magnetic, np.complex128, "magnetic", self.electric.shape),
            )


@dataclass(frozen=True)
class PlaneProvenance:
    """Where a plane came from and what was done to it.

    Attributes
    ----------
    source : str, optional
        The source: a run's configuration hash, ``"pulse"`` or ``"samples"``.
    plane : int or None, optional
        The saved-plane index it was read from, counted from the start.
    z : float or None, optional
        The saved plane's position, m.
    lineage : tuple of str, optional
        Configuration hashes of the runs this one descends from, oldest first.
    history : tuple of str, optional
        The preparation steps applied, in order.
    """

    source: str = "samples"
    plane: int | None = None
    z: float | None = None
    lineage: tuple[str, ...] = ()
    history: tuple[str, ...] = ()

    def then(self, step: str) -> "PlaneProvenance":  # noqa: UP037 - beartype evaluates the class body
        """Return the provenance with one more preparation step.

        Parameters
        ----------
        step : str
            What was done.

        Returns
        -------
        PlaneProvenance
            The same source, with `step` appended to the history.
        """
        return dataclasses.replace(self, history=(*self.history, step))

    def record(self) -> dict[str, object]:
        """Return the JSON-ready record of the provenance.

        Returns
        -------
        dict
            Every field.
        """
        return {
            "source": self.source,
            "plane": self.plane,
            "z": self.z,
            "lineage": list(self.lineage),
            "history": list(self.history),
        }


@dataclass(frozen=True, eq=False)
class FieldPlane:
    """A coherent field at one physical plane, in its native representation.

    Build one with :func:`solprop.processing.field_plane` from a saved run, with
    :meth:`from_pulse` from the package's beam builder, or with :meth:`from_samples`
    from arrays; never by hand unless the parts are already validated.

    Attributes
    ----------
    kind : {'modal', 'general_modal', 'radial', 'cartesian', 'planar'}
        The native representation; the module's table gives each one's columns.
    samples : numpy.ndarray
        Optics-convention raw spectrum of each column referenced to the window
        centre, complex128, shape ``(n_modes, n_pol, n_omega)``, read-only.
    grid : solprop.grid.TimeGrid
        The temporal grid: layout (``field`` or ``envelope``), time samples, the
        absolute frequency of each bin and the carrier (``grid.omega0``).
    units : str
        Units of `samples`: ``"sqrt(W)"``, ``"V m"``, ``"V/m"`` or ``"sqrt(W/m)"``.
    basis : PlaneBasis
        What the columns are, with their identity.
    reference : PlaneReference
        The affine reference of the stored spectrum.
    geometry : PlaneGeometry
        The oriented plane.
    metric : PlaneMetric
        The native power metric at the plane.
    exterior : {'zero', 'unknown'}, optional
        What the record says outside its samples. A run's own planes are zero
        outside their window and grid; a cropped measurement may not be.
    medium : solprop.spatial.SpatialMedium or None, optional
        The bulk medium at a spatial plane, when it is at hand.
    certificate : collections.abc.Mapping or None, optional
        The scoped path certificate of the producing run (SEP 34, section 4.5),
        read-only; it bounds the linear background path, not the whole run.
    nodal : NodalField or None, optional
        Physical E (and H) at transverse nodes, for a general modal plane read with
        its source simulation.
    provenance : PlaneProvenance, optional
        Source identity, plane index, lineage and preparation history.

    Raises
    ------
    CouplingError
        If the samples are not finite and three-dimensional, or disagree with the
        grid or the basis kind.
    """

    kind: PlaneKind
    samples: Complex[np.ndarray, "n_modes n_pol n_omega"]
    grid: TimeGrid
    units: str
    basis: PlaneBasis
    reference: PlaneReference
    geometry: PlaneGeometry
    metric: PlaneMetric
    exterior: Exterior = "zero"
    medium: SpatialMedium | None = None
    certificate: Mapping[str, object] | None = None
    nodal: NodalField | None = None
    provenance: PlaneProvenance = field(default_factory=PlaneProvenance)

    def __post_init__(self) -> None:
        """Own read-only samples and check them against the grid and the basis."""
        samples = _frozen(self.samples, np.complex128, "samples")
        if samples.ndim != 3 or samples.shape[2] != self.grid.n_omega:
            msg = (
                f"samples have shape {samples.shape}; a plane holds (n_modes, n_pol, "
                f"n_omega) with n_omega = {self.grid.n_omega}"
            )
            raise CouplingError(msg)
        if self.basis.kind != self.kind:
            msg = (
                f"a {self.kind} plane needs a {self.kind} basis, got {self.basis.kind}"
            )
            raise CouplingError(msg)
        object.__setattr__(self, "samples", samples)
        if self.certificate is not None:
            object.__setattr__(
                self, "certificate", MappingProxyType(dict(self.certificate))
            )

    def __repr__(self) -> str:
        """Return a one-line description without arrays.

        Returns
        -------
        str
            The kind, shape, layout and position.
        """
        n_modes, n_pol, n_omega = self.samples.shape
        return (
            f"FieldPlane({self.kind}, {n_modes}x{n_pol}x{n_omega} {self.grid.kind}, "
            f"z={self.geometry.z:.6g} m, {self.units})"
        )

    @property
    def omega(self) -> Float[np.ndarray, " n_omega"]:
        """Return the absolute angular frequency of each stored bin.

        Returns
        -------
        numpy.ndarray
            Angular frequencies in rad/s, shape ``(n_omega,)``, in storage order.
        """
        return self.grid.omega

    def absolute(self) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the samples in the absolute reference.

        Returns
        -------
        numpy.ndarray
            ``samples * reference.factor(omega)``, shape ``(n_modes, n_pol,
            n_omega)``.

        Raises
        ------
        CouplingError
            If the reference is not calibrated in both terms.
        """
        if self.reference.calibrated != ("phase", "delay"):
            msg = (
                "this plane's absolute reference is not known in "
                f"{sorted({'phase', 'delay'} - set(self.reference.calibrated))}"
            )
            raise CouplingError(msg)
        return self.samples * self.reference.factor(self.omega)[None, None, :]

    def spectral_energy(self) -> Float[np.ndarray, " n_omega"]:
        """Return the energy each bin carries through the plane, J.

        Returns
        -------
        numpy.ndarray
            :meth:`PlaneMetric.spectral_energy` of the samples, shape
            ``(n_omega,)``.
        """
        return self.metric.spectral_energy(self.samples, self.grid)

    def energy(self) -> float:
        """Return the physical energy crossing the plane, J.

        Returns
        -------
        float
            The sum of :meth:`spectral_energy`; J per metre of y on a line.
        """
        return float(np.sum(self.spectral_energy()))

    def node_spectra(self) -> Complex[np.ndarray, "n_nodes n_pol n_omega"]:
        """Return a spatial plane's electric field at its grid's nodes.

        Returns
        -------
        numpy.ndarray
            Optics-convention raw spectra in V/m, shape ``(n_nodes, n_pol,
            n_omega)``, nodes in the grid's order.

        Raises
        ------
        CouplingError
            For a plane that is not radial or Cartesian.
        """
        space = self.basis.space
        if space is None or self.kind not in ("radial", "cartesian"):
            msg = f"a {self.kind} plane has no node samples of its own"
            raise CouplingError(msg)
        internal = fourier.spectrum_to_internal(self.samples, kind=self.grid.kind)
        return fourier.spectrum_to_optics(
            _transverse(internal, space, inverse=True), kind=self.grid.kind
        )

    @classmethod
    def from_pulse(
        cls,
        pulse: object,
        grid: TimeGrid,
        geometry: PlaneGeometry,
        *,
        space: RadialGrid | FreeGrid,
        medium: SpatialMedium,
        reference: PlaneReference | None = None,
    ) -> "FieldPlane":  # noqa: UP037 - beartype evaluates the class body
        """Build a spatial plane from the package's beam builder.

        The beam is :meth:`solprop.pulses.SpatialPulse.build` on the grid, with its
        requested energy or peak power the physical axial flux through the plane in
        the medium there (SEP 34); no second Gaussian is written. Its exterior is
        zero: the closed-form beam is resolved on the grid's whole window.

        Parameters
        ----------
        pulse : solprop.pulses.SpatialPulse
            The beam.
        grid : solprop.grid.TimeGrid
            Temporal grid.
        geometry : PlaneGeometry
            The plane; its ``z`` and ``side`` select the medium's sample.
        space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
            Transverse grid.
        medium : solprop.spatial.SpatialMedium
            The medium at the plane.
        reference : PlaneReference or None, optional
            The reference of the built spectrum; default zero phase and delay.

        Returns
        -------
        FieldPlane
            A radial or Cartesian plane with its admittance metric.

        Raises
        ------
        CouplingError
            If `pulse` has no spatial ``build``.
        """
        build = getattr(pulse, "build", None)
        if build is None or not callable(build):
            msg = "from_pulse takes a SpatialPulse"
            raise CouplingError(msg)
        spectrum = cast(
            "np.ndarray",
            build(grid, space, medium=medium, z=geometry.z, side=geometry.side),
        )
        flux = SpatialFlux.resolve(
            grid, space, medium, z=geometry.z, side=geometry.side
        )
        basis = PlaneBasis.of_space(space)
        return cls(
            basis.kind,
            fourier.spectrum_to_optics(spectrum, kind=grid.kind),
            grid,
            "V m" if basis.kind == "radial" else "V/m",
            basis,
            PlaneReference() if reference is None else reference,
            geometry,
            PlaneMetric.from_flux(flux),
            "zero",
            medium,
            provenance=PlaneProvenance("pulse", history=(repr(pulse),)),
        )

    @classmethod
    def from_samples(  # noqa: PLR0913 - the arrays and every declaration they need
        cls,
        samples: Complex[np.ndarray, "n_nodes n_pol n_omega"],
        grid: TimeGrid,
        geometry: PlaneGeometry,
        *,
        exterior: Exterior,
        space: RadialGrid | FreeGrid | None = None,
        medium: SpatialMedium | None = None,
        basis: PlaneBasis | None = None,
        metric: PlaneMetric | None = None,
        reference: PlaneReference | None = None,
    ) -> "FieldPlane":  # noqa: UP037 - beartype evaluates the class body
        """Bind physical arrays to explicit axes, units, domain, medium and reference.

        Two forms. With `space` and `medium`, `samples` is the electric field at the
        grid's nodes: optics-convention raw temporal spectra in V/m, referenced to
        the window centre, shape ``(n_nodes, n_pol, n_omega)``; they are transformed
        to the grid's native columns and measured with the medium's axial
        admittance. With `basis` (a modal basis), `samples` is already the
        optics-convention coefficient spectrum in sqrt(W), shape ``(n_modes,
        n_pol, n_omega)``, measured with `metric` (the identity by default).

        Parameters
        ----------
        samples : numpy.ndarray
            The arrays, as above.
        grid : solprop.grid.TimeGrid
            Temporal grid of the samples.
        geometry : PlaneGeometry
            The plane.
        exterior : {'zero', 'unknown'}
            Whether the field is declared zero outside the samples, or unknown
            there. There is no default: a crop of a measured beam is not a
            complete beam.
        space : solprop.grid.RadialGrid or solprop.grid.FreeGrid or None, optional
            Transverse grid of node samples.
        medium : solprop.spatial.SpatialMedium or None, optional
            Medium at the plane, required with `space`.
        basis : PlaneBasis or None, optional
            Modal basis of coefficient samples, instead of `space`.
        metric : PlaneMetric or None, optional
            Metric of a modal basis; default the identity.
        reference : PlaneReference or None, optional
            Reference of the spectrum; default zero phase and delay.

        Returns
        -------
        FieldPlane
            The plane.

        Raises
        ------
        CouplingError
            If neither or both forms are given, the shapes disagree, or a spatial
            form has no medium.
        """
        values = np.asarray(samples, dtype=np.complex128)
        reference = PlaneReference() if reference is None else reference
        provenance = PlaneProvenance("samples")
        if (space is None) == (basis is None):
            msg = (
                "from_samples takes node samples on a space or coefficients on a basis"
            )
            raise CouplingError(msg)
        if space is not None:
            if medium is None:
                msg = "node samples are measured in their medium: give medium="
                raise CouplingError(msg)
            n_nodes = space.r.size
            if values.ndim != 3 or values.shape[0] != n_nodes:
                msg = f"node samples need {n_nodes} nodes, got shape {values.shape}"
                raise CouplingError(msg)
            internal = fourier.spectrum_to_internal(values, kind=grid.kind)
            columns = fourier.spectrum_to_optics(
                _transverse(internal, space, inverse=False), kind=grid.kind
            )
            flux = SpatialFlux.resolve(
                grid, space, medium, z=geometry.z, side=geometry.side
            )
            plane_basis = PlaneBasis.of_space(space)
            return cls(
                plane_basis.kind,
                columns,
                grid,
                "V m" if plane_basis.kind == "radial" else "V/m",
                plane_basis,
                reference,
                geometry,
                PlaneMetric.from_flux(flux),
                exterior,
                medium,
                provenance=provenance,
            )
        assert basis is not None
        if basis.kind not in ("modal", "general_modal"):
            msg = f"coefficient samples need a modal basis, got {basis.kind}"
            raise CouplingError(msg)
        return cls(
            basis.kind,
            values,
            grid,
            "sqrt(W)",
            basis,
            reference,
            geometry,
            PlaneMetric("identity") if metric is None else metric,
            exterior,
            provenance=provenance,
        )


def _transverse(
    values: Complex[np.ndarray, "n_columns n_pol n_omega"],
    space: RadialGrid | FreeGrid,
    *,
    inverse: bool,
) -> Complex[np.ndarray, "n_columns n_pol n_omega"]:
    """Apply a grid's transverse transform along the column axis.

    Parameters
    ----------
    values : numpy.ndarray
        Internal-convention spectra, shape ``(n_columns, n_pol, n_omega)``.
    space : solprop.grid.RadialGrid or solprop.grid.FreeGrid
        The grid whose QDHT or two-dimensional FFT is applied.
    inverse : bool
        True for columns to nodes, False for nodes to columns.

    Returns
    -------
    numpy.ndarray
        The transformed spectra, same shape.
    """
    if isinstance(space, RadialGrid):
        moved = np.moveaxis(values, 0, -1)
        result = space.hankel.inverse(moved) if inverse else space.hankel.forward(moved)
        return np.moveaxis(np.asarray(result), -1, 0)
    shaped = values.reshape(space.n_y, space.n_x, *values.shape[1:])
    result = (
        np.fft.ifft2(shaped, axes=(0, 1))
        if inverse
        else np.fft.fft2(shaped, axes=(0, 1))
    )
    return result.reshape(values.shape)

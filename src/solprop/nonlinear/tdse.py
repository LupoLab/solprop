# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Declare the immutable, provider-independent helium TDSE response specification.

All physical quantities use SI. The optional atomic package is imported only by
its runtime adapter, never by these records. See SEP 12 for the distinct linear,
cubic and ionisation calibrations and the restricted first implementation, and
SEP 20 for the haCC two-electron model (:class:`HaccBasis`), the atomic time
window (:class:`TDSEAtomicWindow`) and the weak-field consistency policy
(:class:`TDSEConsistency`).
"""

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import KW_ONLY, asdict, dataclass, field
from typing import Literal

from solprop.constants import AU_TIME
from solprop.exceptions import MaterialError
from solprop.nonlinear.responses import ResponseTraits

__all__ = [
    "HaccBasis",
    "TDSEAtomicWindow",
    "TDSEBasis",
    "TDSECalibration",
    "TDSEConsistency",
    "TDSEExecution",
    "TDSEGasResponse",
    "TDSEKerrModel",
    "TDSEModel",
    "TDSESampling",
]

TDSEKerrModel = Literal["optmat_cubic", "tdse_shadow", "tdse_native"]
"""The three explicit cubic retention or replacement policies of SEP 12."""

TDSEModel = Literal["sae", "hacc"]
"""The atomic model family: single active electron, or two-electron haCC (SEP 20)."""

_MAX_ANGULAR_ORDER = 4096
_MAX_RADIAL_ORDER = 64
_MAX_HACC_ANGULAR = 120
_MAX_NEUTRAL_SQUARE = 7
# The provider's preset identity: model number (1 for haCC(1), 5 for haCC(5)), the
# preset-definition version and the first 16 hex digits of its SHA-256.
# The provider's preset identities: ``hacc-helium-<1|5>-v<n>-<digest>`` for the
# standard numerics and ``hacc-helium-<1|5>-fast-v<n>-<digest>`` for the fast variant
# (cut 125 hartree, RK4 step 0.02045 au); the variant is part of the identity.
_PRESET_IDENTITY = re.compile(
    r"hacc-helium-([15])(?:-(fast))?-v([1-9][0-9]*)-([0-9a-f]{16})"
)


def _positive(value: float, name: str) -> None:
    """Require a finite positive real scalar, excluding booleans.

    Parameters
    ----------
    value : float
        Candidate scalar in the calling field's declared SI units.
    name : str
        Field name used in the diagnostic.
    """
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise MaterialError(f"{name} must be positive and finite")


def _count(value: int, name: str, minimum: int = 1) -> None:
    """Require a plain integer above the declared lower bound.

    Parameters
    ----------
    value : int
        Candidate dimensionless count.
    name : str
        Field name used in the diagnostic.
    minimum : int, optional
        Inclusive lower bound, default 1.
    """
    if type(value) is not int or value < minimum:
        raise MaterialError(f"{name} must be an integer >= {minimum}")


def _text(value: str, name: str) -> None:
    """Require a nonempty identifier or source description.

    Parameters
    ----------
    value : str
        Candidate text, which must contain a non-whitespace character.
    name : str
        Field name used in the diagnostic.
    """
    if not isinstance(value, str) or not value.strip():
        raise MaterialError(f"{name} must be a nonempty string")


def _strings(values: tuple[str, ...], name: str) -> None:
    """Keep sequence metadata immutable and free of empty entries.

    Parameters
    ----------
    values : tuple of str
        Candidate immutable nonempty strings; the tuple itself may be empty.
    name : str
        Field name used in the diagnostic.
    """
    if type(values) is not tuple:
        raise MaterialError(f"{name} must be a tuple")
    for value in values:
        _text(value, name)


@dataclass(frozen=True)
class TDSEBasis:
    """Describe a fully resolved helium atomic basis without native objects.

    Attributes
    ----------
    name : str
        Descriptive preset or custom-basis name; numerical fields are authoritative.
    radial_edges : tuple of float
        Increasing finite-element boundaries in m, beginning at zero.
    radial_orders : tuple of int
        Gauss-Lobatto points per finite element, each between 2 and 64.
    lmax : int
        Largest angular momentum, nonnegative and at most 4096.
    cut_energy : float
        Positive spectral cut in J.
    source : str
        Provenance for the numerical choices.
    m : int
        Magnetic quantum number; the initial helium route requires zero.
    potential : str
        Built-in provider identity, initially ``helium_tong05`` only.
    potential_version : int
        Positive revision of the provider's potential definition.
    scaling_radius : float or None
        Exterior-complex-scaling boundary in m, exactly a positive radial edge.
    scaling_angle : float or None
        Scaling angle in rad, strictly between zero and pi/2.
    infinite_order : int or None
        Infinite-element points, between 1 and 64.
    decay : float or None
        Positive infinite-element decay exponent in 1/m.
    cuts : tuple of float
        Optional observation boundaries in m, increasing positive edges below
        the physical boundary. Scaling fields must all be given or all absent.
    """

    name: str
    radial_edges: tuple[float, ...]
    radial_orders: tuple[int, ...]
    lmax: int
    cut_energy: float
    _: KW_ONLY
    source: str
    m: int = 0
    potential: str = "helium_tong05"
    potential_version: int = 1
    scaling_radius: float | None = None
    scaling_angle: float | None = None
    infinite_order: int | None = None
    decay: float | None = None
    cuts: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Validate the immutable radial, angular and model description."""
        _text(self.name, "basis name")
        _text(self.source, "basis source")
        if self.potential != "helium_tong05":
            raise MaterialError("TDSE initially supports helium_tong05 only")
        _count(self.potential_version, "potential_version")
        _count(self.lmax, "lmax", 0)
        if self.lmax > _MAX_ANGULAR_ORDER or type(self.m) is not int or self.m != 0:
            raise MaterialError("TDSE helium requires m=0 and lmax <= 4096")
        _positive(self.cut_energy, "cut_energy")
        self._validate_radial()
        self._validate_scaling()
        boundary = (
            self.radial_edges[-1]
            if self.scaling_radius is None
            else self.scaling_radius
        )
        if type(self.cuts) is not tuple:
            raise MaterialError("cuts must be a tuple")
        previous = 0.0
        for cut in self.cuts:
            _positive(cut, "cut")
            if cut not in self.radial_edges or not previous < cut < boundary:
                raise MaterialError(
                    "cuts must be increasing radial edges inside the boundary"
                )
            previous = cut
        object.__setattr__(self, "radial_edges", tuple(map(float, self.radial_edges)))
        object.__setattr__(self, "cuts", tuple(map(float, self.cuts)))
        object.__setattr__(self, "cut_energy", float(self.cut_energy))
        for name in ("scaling_radius", "scaling_angle", "decay"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))

    def _validate_radial(self) -> None:
        """Check element sizes before constructing any provider basis."""
        _radial_axis(self.radial_edges, self.radial_orders)

    def _validate_scaling(self) -> None:
        """Keep the optional infinite element a complete explicit specification."""
        values = (
            self.scaling_radius,
            self.scaling_angle,
            self.infinite_order,
            self.decay,
        )
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise MaterialError("all four scaling fields must be given together")
        assert self.scaling_radius is not None and self.scaling_angle is not None
        assert self.infinite_order is not None and self.decay is not None
        _positive(self.scaling_radius, "scaling_radius")
        _positive(self.scaling_angle, "scaling_angle")
        _positive(self.decay, "decay")
        _count(self.infinite_order, "infinite_order")
        if self.scaling_radius not in self.radial_edges:
            raise MaterialError("scaling_radius must exactly equal a radial edge")
        if self.scaling_angle >= math.pi / 2 or self.infinite_order > _MAX_RADIAL_ORDER:
            raise MaterialError(
                "scaling requires angle < pi/2 and infinite_order <= 64"
            )

    @property
    def identity(self) -> str:
        """Return the stable SHA-256 identity of all resolved basis fields.

        Returns
        -------
        str
            Hexadecimal digest of canonical JSON, including the source record.
        """
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


def _radial_axis(edges: Sequence[float], orders: Sequence[int]) -> None:
    """Validate an immutable finite-element axis starting at zero.

    Parameters
    ----------
    edges : sequence of float
        Candidate element boundaries in m, beginning with zero; must be a tuple.
    orders : sequence of int
        Candidate points per element, each between 2 and 64; must be a tuple.
    """
    if type(edges) is not tuple or type(orders) is not tuple:
        raise MaterialError("radial_edges and radial_orders must be tuples")
    if len(edges) < 2 or len(orders) != len(edges) - 1:
        raise MaterialError(
            "radial_edges must contain one more entry than radial_orders"
        )
    if isinstance(edges[0], bool) or edges[0] != 0:
        raise MaterialError("the first radial edge must be zero")
    previous = 0.0
    for edge in edges[1:]:
        _positive(edge, "radial edge")
        if edge <= previous:
            raise MaterialError("radial_edges must be strictly increasing")
        previous = edge
    for order in orders:
        _count(order, "radial order", 2)
        if order > _MAX_RADIAL_ORDER:
            raise MaterialError("radial orders must be <= 64")


@dataclass(frozen=True)
class HaccBasis:
    """Describe a helium haCC model and basis, a provider preset or a custom one.

    haCC (hybrid anti-symmetrised coupled channels; Majety, Zielinski and Scrinzi,
    New J. Phys. 17, 063002, 2015, doi:10.1088/1367-2630/17/6/063002) expands the
    two-electron state as a correlated neutral plus antisymmetrised products of frozen
    He+ channels with an active electron on a radial finite-element axis absorbed by
    infinite-range exterior complex scaling beyond ``scaling_radius``. Every field is
    SI except ``max_step_au``, an explicitly named exception that preserves the
    provider's exact binary step ceiling in atomic units of time.

    Attributes
    ----------
    ions : {1, 2}
        Frozen He+ shells ``n <= ions``: 1 is haCC(1) (the 1s channel), 2 is haCC(5)
        (1s, 2s, 2p0 and 2p+-1).
    neutral_square : int
        Largest ``l`` of the correlated neutral's ``(l, l, 0)`` squares, 0 to 7.
    radial_edges : tuple of float
        Element boundaries in m, from zero, strictly increasing.
    radial_orders : tuple of int
        Points per element, each 2 to 64.
    angular_limit : {'particle', 'total'}
        Whether ``angular_order`` bounds the active electron's ``l`` or the total ``L``.
    angular_order : int
        The angular bound, 1 to 120.
    scaling_radius : float
        R0 in m, exactly a nonzero radial edge, where irECS starts.
    scaling_angle : float
        irECS angle in rad, strictly between 0 and pi/2.
    infinite_order : int
        Points of the infinite element, 1 to 64.
    decay : float
        Positive infinite-element decay exponent in 1/m.
    gauge_radius : float or None
        Rg in m, a radial edge strictly inside R0: length gauge inside, velocity gauge
        outside (the mixed gauge). ``None`` is the pure velocity-gauge control,
        representable only for a custom basis.
    cut_energy : float
        Positive spectral cut above the ground root in J.
    max_step_au : float
        Positive largest RK4 step in atomic units of time.
    source : str
        Provenance: the preset's evidence, or the user's reasons for a custom basis.
    preset : str or None
        The provider's preset identity, ``hacc-helium-<1|5>-v<version>-<16 hex>`` for
        the standard numerics or ``hacc-helium-<1|5>-fast-v<version>-<16 hex>`` for
        the fast variant, or ``None`` for a custom basis. Its model number must agree
        with ``ions``.

    Raises
    ------
    MaterialError
        If a field is out of range, of the wrong type or inconsistent.

    Examples
    --------
    >>> from solprop.nonlinear.tdse import HaccBasis
    >>> a0 = 5.29177210544e-11
    >>> basis = HaccBasis(
    ...     1, 1, tuple(x * a0 for x in (0.0, 1.0, 4.0, 12.0)), (6, 6, 6),
    ...     "particle", 2, 12 * a0, 0.15, 6, 1.0 / a0, 4 * a0, 1e-16, 0.05,
    ...     source="tiny unconverged custom basis",
    ... )
    >>> basis.label, basis.qualified
    ('haCC(1) (custom)', False)
    """

    ions: Literal[1, 2]
    neutral_square: int
    radial_edges: tuple[float, ...]
    radial_orders: tuple[int, ...]
    angular_limit: Literal["particle", "total"]
    angular_order: int
    scaling_radius: float
    scaling_angle: float
    infinite_order: int
    decay: float
    gauge_radius: float | None
    cut_energy: float
    max_step_au: float
    _: KW_ONLY
    source: str
    preset: str | None = None

    def __post_init__(self) -> None:
        """Validate the model, axis, absorber, gauge and preset identity."""
        if type(self.ions) is not int or self.ions not in (1, 2):
            raise MaterialError("HaccBasis ions must be 1 (haCC(1)) or 2 (haCC(5))")
        _count(self.neutral_square, "neutral_square", 0)
        if self.neutral_square > _MAX_NEUTRAL_SQUARE:
            raise MaterialError("neutral_square must be <= 7")
        _radial_axis(self.radial_edges, self.radial_orders)
        if self.angular_limit not in ("particle", "total"):
            raise MaterialError("angular_limit must be particle or total")
        _count(self.angular_order, "angular_order")
        if self.angular_order > _MAX_HACC_ANGULAR:
            raise MaterialError("angular_order must be <= 120")
        self._validate_scaling()
        if self.gauge_radius is not None:
            _positive(self.gauge_radius, "gauge_radius")
            if (
                self.gauge_radius not in self.radial_edges
                or not self.gauge_radius < self.scaling_radius
            ):
                raise MaterialError(
                    "gauge_radius must be a radial edge strictly inside scaling_radius"
                )
        _positive(self.cut_energy, "cut_energy")
        _positive(self.max_step_au, "max_step_au")
        _text(self.source, "basis source")
        if self.preset is not None:
            match = (
                _PRESET_IDENTITY.fullmatch(self.preset)
                if isinstance(self.preset, str)
                else None
            )
            if match is None:
                raise MaterialError(
                    "preset must be a provider identity "
                    "hacc-helium-<1|5>[-fast]-v<version>-<16 hex digits>"
                )
            if int(match.group(1)) != (1 if self.ions == 1 else 5):
                raise MaterialError("preset model number disagrees with ions")
            if self.gauge_radius is None:
                raise MaterialError("a preset haCC basis requires a gauge radius")
        object.__setattr__(self, "radial_edges", tuple(map(float, self.radial_edges)))
        for name in (
            "scaling_radius",
            "scaling_angle",
            "decay",
            "cut_energy",
            "max_step_au",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.gauge_radius is not None:
            object.__setattr__(self, "gauge_radius", float(self.gauge_radius))

    def _validate_scaling(self) -> None:
        """Require a complete exterior-scaling specification at a radial edge."""
        values = (
            self.scaling_radius,
            self.scaling_angle,
            self.infinite_order,
            self.decay,
        )
        if any(value is None for value in values):
            raise MaterialError("all four irECS fields must be given")
        _positive(self.scaling_radius, "scaling_radius")
        _positive(self.scaling_angle, "scaling_angle")
        _positive(self.decay, "decay")
        _count(self.infinite_order, "infinite_order")
        if self.scaling_radius not in self.radial_edges[1:]:
            raise MaterialError("scaling_radius must exactly equal a radial edge")
        if self.scaling_angle >= math.pi / 2 or self.infinite_order > _MAX_RADIAL_ORDER:
            raise MaterialError(
                "scaling requires angle < pi/2 and infinite_order <= 64"
            )

    @property
    def identity(self) -> str:
        """Return the SHA-256 identity of every field, tagged as a haCC basis.

        The ``kind`` tag keeps a haCC identity distinct from any SAE
        :class:`TDSEBasis` identity; ``preset`` distinguishes a preset from a custom
        basis with the same numbers.

        Returns
        -------
        str
            Hexadecimal digest of canonical JSON with ``kind="hacc_helium"``.
        """
        encoded = json.dumps(
            {"kind": "hacc_helium", **asdict(self)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def label(self) -> str:
        """Return the model's name, with ``(custom)`` for a non-preset basis.

        Returns
        -------
        str
            ``haCC(1)`` or ``haCC(5)``, followed by `` fast`` for the fast preset
            variant and by `` (custom)`` when ``preset`` is ``None``.
        """
        name = "haCC(1)" if self.ions == 1 else "haCC(5)"
        if self.preset is None:
            return f"{name} (custom)"
        return name if self.variant == "standard" else f"{name} fast"

    @property
    def variant(self) -> Literal["standard", "fast"] | None:
        """Return the preset's numerics variant, read from its identity.

        Returns
        -------
        {'standard', 'fast'} or None
            ``'fast'`` for a ``hacc-helium-<1|5>-fast-...`` identity, ``'standard'``
            for any other preset, ``None`` for a custom basis.
        """
        if self.preset is None:
            return None
        match = _PRESET_IDENTITY.fullmatch(self.preset)
        assert match is not None  # validated at construction
        return "fast" if match.group(2) else "standard"

    @property
    def qualified(self) -> bool:
        """Return whether the record claims a provider preset's qualification.

        The claim is verified against the installed provider when the binding
        prepares the model; a custom basis is never qualified.

        Returns
        -------
        bool
            True exactly when ``preset`` is set.
        """
        return self.preset is not None


@dataclass(frozen=True)
class TDSEAtomicWindow:
    """Drive the atom only over an interval of the optical lattice (SEP 20, 4.6).

    The atom starts from its ground state at the first lattice sample at or after
    ``-duration / 2`` and stops at the last at or before ``+duration / 2``. The drive
    and the nonlinear source are both multiplied by the same gate: a Planck taper of
    width ``taper`` inside each end, one between the tapers and zero outside. The gate
    is the source gate of the windowed response, so a window needs the native closure,
    whose source is gated. Dropping the field
    outside the window is a finite-memory approximation, not a property of FFT
    padding.

    Attributes
    ----------
    duration : float or None
        Nominal atomic duration in s, finite and positive. ``None`` is resolved from
        the capillary request's ``trange``; a direct construction must give it.
    taper : float
        Width of each inner Planck taper in s, finite and positive; it must be less
        than half the duration.

    Raises
    ------
    MaterialError
        If a value is non-finite, non-positive, boolean, or the taper is not shorter
        than half the duration.
    """

    duration: float | None = None
    taper: float = 5e-15

    def __post_init__(self) -> None:
        """Validate the duration and taper before any grid is known."""
        _positive(self.taper, "atomic window taper")
        object.__setattr__(self, "taper", float(self.taper))
        if self.duration is not None:
            _positive(self.duration, "atomic window duration")
            object.__setattr__(self, "duration", float(self.duration))
            if not self.taper < self.duration / 2:
                raise MaterialError(
                    "atomic window taper must be less than half its duration"
                )


@dataclass(frozen=True)
class TDSEConsistency:
    """Set the policy of the haCC weak-field linear consistency check (SEP 20, 5.4).

    Before the first propagation call of a haCC run the per-atom linear
    susceptibility of the model is compared with optmat's helium through the ratio
    kappa_1. ``|kappa_1 - 1| <= warn`` passes, up to ``refuse`` warns, above it
    refuses; an invalid extraction always refuses. These are policy, not physics:
    they never enter a descriptor, identity or resume key.

    Attributes
    ----------
    warn : float
        Positive finite warning bound on ``|kappa_1 - 1|``; default 0.1.
    refuse : float
        Finite refusal bound, at least ``warn``; default 0.3, the range of gross
        errors (an electron count applied twice, a sign, a unit).
    enforce : bool
        False skips the check entirely, with a warning: no atomic call, no cache.

    Raises
    ------
    MaterialError
        If a bound is boolean, non-finite, non-positive or ``warn > refuse``.
    """

    warn: float = 0.1
    refuse: float = 0.3
    enforce: bool = True

    def __post_init__(self) -> None:
        """Validate the bounds and the switch."""
        _positive(self.warn, "consistency warn")
        _positive(self.refuse, "consistency refuse")
        if self.warn > self.refuse:
            raise MaterialError("consistency warn must not exceed refuse")
        if type(self.enforce) is not bool:
            raise MaterialError("consistency enforce must be a bool")
        object.__setattr__(self, "warn", float(self.warn))
        object.__setattr__(self, "refuse", float(self.refuse))


@dataclass(frozen=True)
class TDSECalibration:
    """Record the qualification status and domain of one calibration.

    Attributes
    ----------
    identity : str
        Persistent identity of the measured or explicitly provisional record.
    status : {'research', 'qualified'}
        Whether the declared domain has supporting qualification evidence.
    basis_identity : str
        Identity of the resolved basis used for that record.
    wavelength_range : tuple of float
        Increasing positive wavelength bounds in m; these describe the stated
        calibration, not arbitrary broadband frequency mixing.
    source : str
        Evidence reference, including the source of a research hypothesis.
    uncertainty : float or None
        Nonnegative finite relative uncertainty; required for a qualified record.
    limitations : tuple of str
        Explicit interpretation limits; a research record requires at least one.
    """

    identity: str
    status: Literal["research", "qualified"]
    basis_identity: str
    wavelength_range: tuple[float, float]
    _: KW_ONLY
    source: str
    uncertainty: float | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a qualified label without its domain, identity and uncertainty."""
        for name in ("identity", "basis_identity", "source"):
            _text(getattr(self, name), name)
        if self.status not in ("research", "qualified"):
            raise MaterialError("calibration status must be research or qualified")
        if type(self.wavelength_range) is not tuple or len(self.wavelength_range) != 2:
            raise MaterialError("wavelength_range must be a pair")
        for wavelength in self.wavelength_range:
            _positive(wavelength, "calibration wavelength")
        if self.wavelength_range[0] >= self.wavelength_range[1]:
            raise MaterialError("wavelength_range must be increasing")
        _strings(self.limitations, "limitations")
        if self.uncertainty is not None and (
            isinstance(self.uncertainty, bool)
            or not math.isfinite(self.uncertainty)
            or self.uncertainty < 0
        ):
            raise MaterialError(
                "calibration uncertainty must be nonnegative and finite"
            )
        if self.status == "qualified" and self.uncertainty is None:
            raise MaterialError("qualified calibration requires uncertainty")
        if self.status == "research" and not self.limitations:
            raise MaterialError("research calibration requires explicit limitations")


@dataclass(frozen=True)
class TDSESampling:
    """Specify integer refinement of the existing fine response lattice.

    Attributes
    ----------
    refinement : int
        Positive power of two multiplying the existing fine sample count.
        The coarse spectrum, time window and collars stay fixed. No tail taper,
        endpoint duplication or field-area correction is applied.
    """

    refinement: int = 1

    def __post_init__(self) -> None:
        """Admit only refinements that preserve the existing binary lattice."""
        _count(self.refinement, "refinement")
        if self.refinement & (self.refinement - 1):
            raise MaterialError("refinement must be a power of two")


@dataclass(frozen=True)
class TDSEExecution:
    """Declare runtime resources separately from the atomic scientific model.

    Attributes
    ----------
    device : {'cpu', 'cuda', 'hip'}
        Required installed provider execution family; there is no implicit fallback.
    memory_limit : int or None
        Positive persistent-memory budget in bytes, or no explicit limit.
    max_batch_rows : int
        Positive maximum number of radial rows in one provider call.
    max_atomic_calls : int or None
        Positive limit on attempted provider calls, including failures and diagnostics.
    complement : {'deterministic', 'vendor_blas'}
        Explicit provider projector arithmetic; availability is checked at preparation.
    """

    device: Literal["cpu", "cuda", "hip"] = "cpu"
    memory_limit: int | None = None
    max_batch_rows: int = 32
    max_atomic_calls: int | None = None
    complement: Literal["deterministic", "vendor_blas"] = "deterministic"

    def __post_init__(self) -> None:
        """Validate resource limits without importing or initializing a runtime."""
        if self.device not in ("cpu", "cuda", "hip"):
            raise MaterialError("TDSE device must be cpu, cuda or hip")
        if self.complement not in ("deterministic", "vendor_blas"):
            raise MaterialError("TDSE complement must be deterministic or vendor_blas")
        _count(self.max_batch_rows, "max_batch_rows")
        for name in ("memory_limit", "max_atomic_calls"):
            value = getattr(self, name)
            if value is not None:
                _count(value, name)


@dataclass(frozen=True)
class TDSEGasResponse:
    """Own the complete electronic nonlinear response of scalar helium.

    Attributes
    ----------
    target : str
        Initially ``helium`` only, in its prepared ground state.
    basis : TDSEBasis or HaccBasis
        Fully resolved atomic specification: an SAE radial, angular, spectral-cut and
        potential record for ``model="sae"``, a haCC model and basis for
        ``model="hacc"``.
    max_atomic_step : float
        Positive maximum internal atomic step in s; actual subdivision is recorded.
        For haCC it must equal ``basis.max_step_au`` in seconds within two units in
        the last place; the provider receives ``basis.max_step_au`` itself.
    kerr_model : {'optmat_cubic', 'tdse_shadow', 'tdse_native'}
        Cubic replacement policy. The native option retains the complete native
        nonlinear current after first-order subtraction, with explicit scaling, and
        its source is always gated and edge-free (SEP 26): the remainder ``S`` is
        multiplied by the fine-grid temporal Planck collar, or by the atomic window's
        gate when one is set, and the edge term is subtracted, ``F[g S] - F[g' S] /
        (i omega)`` on the active bins, the transform of the current held at its
        taper-weighted value after the window rather than switched off. The gate
        needs a steady-current certificate of at most 1e-10 on the lattice and band,
        checked at translation. Raw provider currents are unchanged; the cubic
        replacements are never gated.
        haCC admits ``tdse_native`` only: the provider has no third order of its
        closed whole-atom current.
    remainder_scale : float
        Positive finite dimensionless current-remainder scale. Applied after
        subtracting the raw first-order current for ``tdse_native``, or the raw
        first- and third-order currents for the other policies. Native scale 1
        retains the unscaled nonlinearity; any other positive value is explicit.
        haCC requires exactly 1.0: its current is already the whole atom's.
    calibration : TDSECalibration
        Separate evidence identity and domain; never an implicit electron multiplier.
    sampling : TDSESampling
        Response-lattice refinement policy.
    cubic_scale : float or None
        Positive finite dimensionless third-order dipole scale, required only for
        ``tdse_shadow``. It is independent of ``remainder_scale``.
    model : {'sae', 'hacc'}
        Atomic model family, which must match the basis type.
    atomic_window : TDSEAtomicWindow, 'auto' or None
        Interval of the optical lattice the atom is driven over. ``"auto"``
        resolves to ``None`` (the full lattice) for SAE and to
        ``TDSEAtomicWindow()`` (the requested duration, 5 fs tapers) for haCC.
        An explicit ``None`` selects the full lattice for haCC too.
    consistency : TDSEConsistency
        Policy of the haCC weak-field linear check; only the default is admitted
        for SAE. Never part of a descriptor, identity or resume key.
    """

    target: str
    basis: TDSEBasis | HaccBasis
    max_atomic_step: float
    kerr_model: TDSEKerrModel
    remainder_scale: float
    calibration: TDSECalibration
    sampling: TDSESampling
    _: KW_ONLY
    cubic_scale: float | None = None
    model: TDSEModel = "sae"
    atomic_window: TDSEAtomicWindow | Literal["auto"] | None = "auto"
    consistency: TDSEConsistency = field(
        default=TDSEConsistency(), metadata={"record": False}
    )

    def __post_init__(self) -> None:
        """Admit the declared source closures without hidden rescaling."""
        self._validate_records()
        _positive(self.max_atomic_step, "max_atomic_step")
        _positive(self.remainder_scale, "remainder_scale")
        if self.kerr_model not in ("optmat_cubic", "tdse_shadow", "tdse_native"):
            raise MaterialError("unknown TDSE Kerr model")
        if isinstance(self.basis, HaccBasis):
            self._validate_hacc(self.basis)
        elif self.consistency != TDSEConsistency():
            raise MaterialError("the consistency check applies to haCC responses only")
        if self.kerr_model == "tdse_shadow":
            if self.cubic_scale is None:
                raise MaterialError("tdse_shadow requires cubic_scale")
            _positive(self.cubic_scale, "cubic_scale")
        elif self.cubic_scale is not None:
            raise MaterialError("cubic_scale is available only for tdse_shadow")
        self._resolve_window()
        self._require_native_window()
        if isinstance(self.basis, TDSEBasis):
            order = 1 if self.kerr_model == "tdse_native" else 3
            if self.basis.lmax < order:
                raise MaterialError(f"this TDSE closure requires lmax >= {order}")
        if self.calibration.basis_identity != self.basis.identity:
            raise MaterialError(
                "calibration basis_identity does not match the resolved basis"
            )

    def _validate_records(self) -> None:
        """Require the target, model and the record types each model needs."""
        if self.target != "helium":
            raise MaterialError("TDSEGasResponse initially supports helium only")
        if self.model not in ("sae", "hacc"):
            raise MaterialError("TDSE model must be sae or hacc")
        if not isinstance(self.calibration, TDSECalibration):
            raise MaterialError(
                "TDSEGasResponse requires a basis and calibration record"
            )
        expected = TDSEBasis if self.model == "sae" else HaccBasis
        if not isinstance(self.basis, expected):
            if isinstance(self.basis, (TDSEBasis, HaccBasis)):
                raise MaterialError(
                    f"model={self.model!r} requires a {expected.__name__}, "
                    f"not a {type(self.basis).__name__}"
                )
            raise MaterialError(
                "TDSEGasResponse requires a basis and calibration record"
            )
        if not isinstance(self.sampling, TDSESampling):
            raise MaterialError("TDSEGasResponse requires a sampling record")
        if not isinstance(self.consistency, TDSEConsistency):
            raise MaterialError("consistency must be a TDSEConsistency record")

    def _validate_hacc(self, basis: HaccBasis) -> None:
        """Apply the whole-atom rules of SEP 20 section 5.1 to a haCC response.

        Parameters
        ----------
        basis : HaccBasis
            The response's validated haCC basis.
        """
        if self.kerr_model != "tdse_native":
            raise MaterialError(
                f"haCC refuses kerr_model={self.kerr_model!r}: the provider has no "
                "third order of the closed whole-atom current; use tdse_native"
            )
        # Exactly one: the closed current already counts both electrons of one atom
        # (SEP 20 section 4.2). nextafter(1.0, 2.0) is refused as well.
        if self.remainder_scale != 1.0:
            raise MaterialError(
                "haCC requires remainder_scale == 1.0: its current is the whole "
                "atom's, so an electron-count multiplier has no meaning"
            )
        step = basis.max_step_au * AU_TIME
        if abs(self.max_atomic_step - step) > 2 * math.ulp(step):
            raise MaterialError(
                f"haCC max_atomic_step must be basis.max_step_au in seconds ({step!r})"
            )

    def _resolve_window(self) -> None:
        """Resolve the automatic window per model and validate an explicit one."""
        window = self.atomic_window
        if isinstance(window, str):
            if window != "auto":
                raise MaterialError("atomic_window must be a record, 'auto' or None")
            resolved = TDSEAtomicWindow() if self.model == "hacc" else None
            object.__setattr__(self, "atomic_window", resolved)
        elif window is not None and not isinstance(window, TDSEAtomicWindow):
            raise MaterialError("atomic_window must be a record, 'auto' or None")

    def _require_native_window(self) -> None:
        """Require the native closure of a window, whose gate is the source gate."""
        if self.atomic_window is not None and self.kerr_model != "tdse_native":
            raise MaterialError(
                "an atomic window requires kerr_model='tdse_native': the window's "
                "gate is the gate of the native source"
            )

    @property
    def traits(self) -> ResponseTraits:
        """Return the scalar real-field memory response traits.

        Returns
        -------
        ResponseTraits
            Field-grid, scalar, density-dependent temporal response.
        """
        return ResponseTraits("memory", ("field",), "scalar", ("density",))

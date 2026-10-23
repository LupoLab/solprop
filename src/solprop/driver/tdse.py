# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own the optional TDSE runtime behind the provider-free response seam.

The provider's whole-electron currents and shadow dipoles remain unscaled SI
arrays. Source assembly applies the calibration and density separately. Native
imports, persistent plans, resource admission and call accounting live here.

Two atomic models are bound (SEP 12, SEP 20): the single-active-electron (SAE)
response plan, and the helium haCC response plan, either from one of the
provider's qualified presets or from a custom basis. An atomic window drives the
atom over part of the lattice: the binding passes the retained samples multiplied
by the window's gate and pads the returned currents with zeros, so a
:class:`~solprop.translate.tdse.TDSEResult` always spans the full lattice.
"""

import dataclasses
import importlib
import json
import math
import sys
import warnings
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, is_dataclass
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Float

from solprop.constants import AU_TIME, EPS0, C
from solprop.exceptions import (
    BackendError,
    MaterialError,
    MemoryBudgetError,
    SolpropError,
    TDSEArchiveError,
    TDSEDomainWarning,
    TDSEUnqualifiedModelWarning,
)
from solprop.nonlinear.tdse import (
    HaccBasis,
    TDSEBasis,
    TDSECalibration,
    TDSEExecution,
)
from solprop.output.tdse_archive import (
    ArchiveOutputs,
    TDSEArchiveWriter,
    record_serializer,
    writer_memory,
)
from solprop.translate.tdse import (
    HACC_NORM_GROWTH_LIMIT,
    AtomicCallBudget,
    AtomicCallLedger,
    AtomicWindowRecord,
    TDSEBinding,
    TDSEDescriptor,
    TDSEPurpose,
    TDSEResult,
    lattice_times,
    waveform_identity,
)

__all__ = [
    "DiagnosticSession",
    "HaccPreset",
    "basis_from_preset",
    "estimate_binding_memory",
    "hacc_basis",
    "hacc_preset",
    "pinned_revision",
    "prepare_binding",
]

_PROVIDER_REVISION = "876b72d10b5286478125b842ea45548a5051b033"
_Preset = Literal["production", "high_accuracy", "strong_field"]
_HaccVariant = Literal["standard", "fast"]
# Native classes cannot occur in runtime annotations: optional-package-free test
# collection resolves every annotation through beartype. Confine their dynamic
# typing to this adapter; the seam's scientific records and arrays remain typed.
_Native = Any
# The archive role of each evaluation purpose (SEP 20 section 5.8); a predictor or
# corrector of a rejected trial reads back as "rejected".
_ARCHIVE_ROLES: dict[str, str] = {
    "startup": "startup",
    "predictor": "predictor",
    "corrected": "corrector",
    "diagnostic": "diagnostic",
}
# The haCC consumer surface of the provider (SEP 20 section 5.7, C1 to C5).
_HACC_SURFACE = (
    "AngularLimit",
    "Argon",
    "Basis",
    "GridResponse",
    "Helium",
    "MixedGauge",
    "ResponsePlan",
    "TotalCurrentResponse",
    "VelocityGauge",
    "helium_preset",
    "prepare",
)
# A preset's calibration admission band: from 20 eV (hc / 20 eV = 62 nm), the
# presets' below-threshold bound, to 1.6 um, the band the 800 nm qualifying drive and
# run_short's grid occupy (SEP 20 section 5.1). Not an independent qualification.
_PRESET_WAVELENGTH_RANGE = (62e-9, 1.6e-6)


@runtime_checkable
class _Provider(Protocol):
    """Type the optional module at its single dynamic-import boundary."""

    AngularBasis: Callable[..., _Native]
    ComplexScaling: Callable[..., _Native]
    Discretization: Callable[..., _Native]
    DrivenError: type[Exception]
    GridResponse: type
    RadialBasis: Callable[..., _Native]
    ResponsePlan: Callable[..., _Native]
    SpectralOptions: Callable[..., _Native]
    SpectralSetup: Callable[..., _Native]
    StateSelection: Callable[..., _Native]
    TotalCurrentResponse: type
    atomic_model: Callable[[str], _Native]
    build_info: Callable[[], _Native]
    # The haCC namespace is looked up by name (`_hacc_surface`): an SAE-only
    # provider is still a provider, and is refused only when haCC is asked for.
    solve_preset: Callable[[str, _Preset], _Native]
    unit_scale: Callable[[str], _Native]


def pinned_revision() -> str:
    """Return the provider source revision this solprop is pinned to.

    Returns
    -------
    str
        Full Git revision; an installed build must report a prefix of it.
    """
    return _PROVIDER_REVISION


def _load_provider() -> _Provider:
    """Load the optional package only when an atomic operation is requested.

    Returns
    -------
    _Provider
        Loaded public provider API; no native assembly has occurred.
    """
    if sys.version_info[:2] != (3, 14):
        raise BackendError("the pinned TDSE provider requires Python 3.14")
    try:
        return cast("_Provider", importlib.import_module("pyroraptor"))
    except (ImportError, OSError) as error:
        raise BackendError(
            "TDSE response requires the optional pyroraptor dependency; "
            "install the pinned Pyroraptor revision into the environment, "
            "with a matching CPU, CUDA or HIP build"
        ) from error


def _build_record(
    provider: _Provider, execution: TDSEExecution | None
) -> dict[str, object]:
    """Refuse stale native extensions and execution-space mismatches before setup.

    Parameters
    ----------
    provider : _Provider
        Loaded optional provider API.
    execution : TDSEExecution or None
        Required device family, or None when only resolving a preset.

    Returns
    -------
    dict
        Serializable native build and installed distribution identities.
    """
    build = provider.build_info()
    revision = build.source_revision
    if len(revision) < 7 or not _PROVIDER_REVISION.startswith(revision):
        raise BackendError(
            f"TDSE native source {revision!r} does not match pinned "
            f"{_PROVIDER_REVISION}"
        )
    space = build.execution_space.lower()
    devices = {"cpu": {"openmp", "serial", "threads"}, "cuda": {"cuda"}, "hip": {"hip"}}
    if execution is not None and space not in devices[execution.device]:
        raise BackendError(
            f"TDSE requested device={execution.device!r}, but the native build uses "
            f"{build.execution_space!r}"
        )
    record: dict[str, object] = {
        "native": asdict(build),
        "source_pin": _PROVIDER_REVISION,
    }
    try:
        package = distribution("pyroraptor")
    except PackageNotFoundError:
        record["distribution"] = None
    else:
        record["distribution"] = package.version
        source = package.read_text("direct_url.json")
        record["direct_url"] = None if source is None else json.loads(source)
    return record


def _hacc_surface(provider: _Provider) -> _Native:
    """Return the provider's haCC namespace, refusing a build without it.

    Parameters
    ----------
    provider : _Provider
        Loaded optional provider API.

    Returns
    -------
    object
        The ``pyroraptor.hacc`` module.

    Raises
    ------
    BackendError
        If the installed provider has no complete haCC consumer surface.
    """
    hacc = getattr(provider, "hacc", None)
    missing = [name for name in _HACC_SURFACE if not hasattr(hacc, name)]
    if missing:
        raise BackendError(
            "the installed TDSE provider has no haCC consumer surface "
            f"(missing {', '.join(missing)}); install Pyroraptor at the pinned "
            f"revision {_PROVIDER_REVISION}"
        )
    return hacc


def basis_from_preset(target: str, preset: _Preset) -> TDSEBasis:
    """Resolve a provider preset into an immutable, serializable SI basis.

    Parameters
    ----------
    target : str
        Atomic target, currently ``helium`` only.
    preset : {'production', 'high_accuracy', 'strong_field'}
        Name of the pinned provider's measured basis and spectral cut.

    Returns
    -------
    TDSEBasis
        Fully resolved radial, angular, potential and spectral parameters.
        The preset's atomic step is deliberately separate from this basis.

    Raises
    ------
    MaterialError
        If the target or preset is unsupported.
    BackendError
        If the optional provider is missing or its native revision is stale.
    """
    if target != "helium" or preset not in (
        "production",
        "high_accuracy",
        "strong_field",
    ):
        raise MaterialError("TDSE requires helium and a supported provider preset")
    provider = _load_provider()
    _build_record(provider, None)
    selected = provider.solve_preset(target, preset)
    basis = selected.discretization
    scaling = basis.scaling
    return TDSEBasis(
        name=selected.name,
        radial_edges=basis.radial.edges,
        radial_orders=basis.radial.orders,
        lmax=basis.angular.lmax,
        cut_energy=selected.cut_energy,
        source=f"pyroraptor@{_PROVIDER_REVISION}: {selected.source}",
        m=basis.angular.m,
        potential=basis.model.name,
        potential_version=basis.model.version,
        scaling_radius=None if scaling is None else scaling.radius,
        scaling_angle=None if scaling is None else scaling.angle,
        infinite_order=None if scaling is None else scaling.infinite_order,
        decay=None if scaling is None else scaling.decay,
        cuts=basis.cuts,
    )


# ------------------------------------------------------------------- haCC records


def _hacc_fields(
    hacc: _Native,
    atom: _Native,
    basis: _Native,
    gauge: _Native,
    cut_energy: float,
    max_step_au: float,
) -> dict[str, object]:
    """Convert provider haCC objects into the numeric fields of a ``HaccBasis``.

    Parameters
    ----------
    hacc : object
        The provider's haCC namespace.
    atom : object
        ``hacc.Helium``; ``hacc.Argon`` is refused by name.
    basis : object
        ``hacc.Basis``: radial axis, irECS and angular limit, SI.
    gauge : object
        ``hacc.MixedGauge`` or ``hacc.VelocityGauge``.
    cut_energy : float
        Spectral cut in J.
    max_step_au : float
        RK4 step ceiling in atomic units of time.

    Returns
    -------
    dict
        Every ``HaccBasis`` field but ``source`` and ``preset``.

    Raises
    ------
    MaterialError
        If the atom is argon or another non-helium record, the ion count is not 1 or
        2, or a member cannot be represented.
    """
    if isinstance(atom, hacc.Argon):
        raise MaterialError(
            "argon TDCIS is not representable as a HaccBasis: the haCC target is "
            "helium only"
        )
    if not isinstance(atom, hacc.Helium):
        raise MaterialError("a HaccBasis needs a provider hacc.Helium atom")
    if atom.ions not in (1, 2):
        raise MaterialError(
            f"Helium(ions={atom.ions}) is not representable: HaccBasis admits "
            "haCC(1) (ions=1) and haCC(5) (ions=2)"
        )
    if isinstance(gauge, hacc.MixedGauge):
        gauge_radius: float | None = float(gauge.radius)
    elif isinstance(gauge, hacc.VelocityGauge):
        gauge_radius = None
    else:
        raise MaterialError("a HaccBasis gauge must be MixedGauge or VelocityGauge")
    angular = basis.angular
    limit = "particle" if angular.particle is not None else "total"
    scaling = basis.scaling
    return {
        "ions": atom.ions,
        "neutral_square": atom.neutral_square,
        "radial_edges": tuple(float(edge) for edge in basis.radial.edges),
        "radial_orders": tuple(basis.radial.orders),
        "angular_limit": limit,
        "angular_order": angular.particle if limit == "particle" else angular.total,
        "scaling_radius": float(scaling.radius),
        "scaling_angle": float(scaling.angle),
        "infinite_order": scaling.infinite_order,
        "decay": float(scaling.decay),
        "gauge_radius": gauge_radius,
        "cut_energy": float(cut_energy),
        "max_step_au": float(max_step_au),
    }


def _numeric_fields(basis: HaccBasis) -> dict[str, object]:
    """Return a ``HaccBasis``'s fields without its provenance and preset identity.

    Parameters
    ----------
    basis : HaccBasis
        Validated record.

    Returns
    -------
    dict
        The model and numerical fields compared against the provider.
    """
    record = asdict(basis)
    del record["source"], record["preset"]
    return record


def _hacc_objects(
    provider: _Provider, hacc: _Native, basis: HaccBasis
) -> tuple[_Native, _Native, _Native]:
    """Build the provider's atom, basis and gauge from a ``HaccBasis``.

    Parameters
    ----------
    provider : _Provider
        Loaded provider, for its radial basis and complex scaling.
    hacc : object
        Its haCC namespace.
    basis : HaccBasis
        Validated SI record.

    Returns
    -------
    tuple
        ``hacc.Helium``, ``hacc.Basis`` and the gauge.
    """
    angular = (
        hacc.AngularLimit(particle=basis.angular_order)
        if basis.angular_limit == "particle"
        else hacc.AngularLimit(total=basis.angular_order)
    )
    atom = hacc.Helium(ions=basis.ions, neutral_square=basis.neutral_square)
    native = hacc.Basis(
        provider.RadialBasis(basis.radial_edges, basis.radial_orders),
        provider.ComplexScaling(
            basis.scaling_radius,
            basis.scaling_angle,
            basis.infinite_order,
            basis.decay,
        ),
        angular,
    )
    gauge = (
        hacc.VelocityGauge()
        if basis.gauge_radius is None
        else hacc.MixedGauge(basis.gauge_radius)
    )
    return atom, native, gauge


@dataclass(frozen=True)
class HaccPreset:
    """Hold a provider's qualified helium haCC preset as solprop records (SEP 20).

    Attributes
    ----------
    basis : HaccBasis
        Frozen copy of the preset's model and basis, with its provider identity.
    variant : {'standard', 'fast'}
        The provider's numerics variant: ``'standard'`` (cut 200 hartree, RK4 step
        at most 0.0088 au) or ``'fast'`` (cut 125 hartree, step at most 0.02045 au,
        about 2.3 times fewer RK4 steps per call), each with its own identity and
        measured records.
    step_margin : float
        The step ceiling over the largest stable RK4 step the provider measured at
        the preset's cut on the qualifying field: 0.62 (standard) or 0.9 (fast).
    max_atomic_step : float
        Largest RK4 step in s, ``basis.max_step_au`` converted with solprop's atomic
        unit of time; pass it as ``TDSEGasResponse.max_atomic_step``.
    accuracy : float
        Largest relative change of the closed total current, ``max_t |dJ| / max_t
        |J|``, across the preset's numerical ladders (lmax, finite-element order,
        cut, step) on the qualifying waveform, up to 8 fs after its centre. For the
        fast variant, the standard value plus the measured largest fast-to-standard
        change (1.5e-4 of the peak), a triangle-inequality bound.
    box_sensitivity : float
        The same measure across the R0 ladder (80, 120 and 160 bohr).
    gauge_sensitivity : float
        The same measure across the Rg ladder (12, 20 and 28 bohr).
    max_peak_intensity : float
        Largest qualified peak intensity in W/m^2.
    wavelength : float
        Carrier wavelength of the qualifying drive in m.
    window : float
        Atomic-window duration the accuracy ladders drove, in s.
    device_admissible : bool
        Whether the provider's device route admits the preset's lattice.
    identity : str
        The provider's preset identity, equal to ``basis.preset``.
    source : str
        The provider's evidence record.

    Notes
    -----
    The sensitivities are measured on one sampled configuration and waveform; they are
    not error bounds on the helium response or on a propagated spectrum.
    """

    basis: HaccBasis
    variant: _HaccVariant
    step_margin: float
    max_atomic_step: float
    accuracy: float
    box_sensitivity: float
    gauge_sensitivity: float
    max_peak_intensity: float
    wavelength: float
    window: float
    device_admissible: bool
    identity: str
    source: str

    def calibration(self) -> TDSECalibration:
        """Return the research calibration record documenting the preset's evidence.

        Returns
        -------
        TDSECalibration
            Research status, no uncertainty (the sensitivities are not certified
            uncertainties), the 62 nm to 1.6 um admission band, and limitations
            naming the window, the intensity bound, the shadow's finite-tail
            qualification, the diagnostic-only bound and the measured
            sensitivities with their metric and waveform.
        """
        metric = (
            "max_t |dJ| / max_t |J| of the closed total current on the run_short "
            "peak-power waveform (8.0e14 W/cm^2, node 0) up to 8 fs after its centre"
        )
        return TDSECalibration(
            self.identity,
            "research",
            self.basis.identity,
            _PRESET_WAVELENGTH_RANGE,
            source=self.source,
            uncertainty=None,
            limitations=(
                f"qualified over an atomic window of {self.window:.4g} s; longer "
                "windows accumulate about 1% of the peak per escape burst",
                f"qualified up to a peak intensity of {self.max_peak_intensity:.4g} "
                f"W/m^2 at a {self.wavelength:.4g} m carrier, spectrum below the "
                "helium threshold",
                "the order-one shadow equals the closed current's first order only up "
                "to the states' amplitudes at the closure surface",
                "neglected_current_bound is a loose diagnostic, not an accuracy bound",
                f"accuracy {self.accuracy:.4g}, box sensitivity "
                f"{self.box_sensitivity:.4g}, gauge sensitivity "
                f"{self.gauge_sensitivity:.4g}: {metric}; not error bounds",
                f"numerics variant {self.variant!r}: RK4 step at "
                f"{self.step_margin:.2g} of the provider's measured stability limit "
                "at the preset's cut; a stronger or different drive can move it",
                "wavelength_range is an admission band, not independent qualification",
            ),
        )


def hacc_preset(ions: Literal[1, 2], variant: _HaccVariant = "standard") -> HaccPreset:
    """Resolve one of the provider's qualified helium haCC presets.

    Parameters
    ----------
    ions : {1, 2}
        1 for haCC(1) (the He+ 1s channel), 2 for haCC(5) (He+ n <= 2).
    variant : {'standard', 'fast'}, optional
        The provider's numerics variant (``hacc.helium_preset(ions, variant)``).
        ``'fast'`` lowers the cut to 125 hartree and raises the RK4 step ceiling
        to 0.02045 au, with the same lattice, box, gauge and validity domain: about
        2.3 times fewer RK4 steps per call, and a closed current within 1.5e-4 of
        the standard variant's peak on the qualifying field. It has its own
        identity, so its own consistency cache entry.

    Returns
    -------
    HaccPreset
        SI records of the preset's model, accuracy and validity domain. No model
        is prepared.

    Raises
    ------
    MaterialError
        If ``ions`` is not the integer 1 or 2, or ``variant`` is not ``'standard'``
        or ``'fast'``.
    BackendError
        If the optional provider is missing, stale or has no haCC surface, or its
        preset's variant disagrees with the request.

    Examples
    --------
    >>> from solprop.driver.tdse import hacc_preset
    >>> preset = hacc_preset(1, "fast")  # doctest: +SKIP
    >>> preset.basis.label, preset.basis.preset  # doctest: +SKIP
    ('haCC(1) fast', 'hacc-helium-1-fast-v1-e38292e1b30729c1')
    """
    if type(ions) is not int or ions not in (1, 2):
        raise MaterialError("hacc_preset needs ions=1 (haCC(1)) or ions=2 (haCC(5))")
    if not isinstance(variant, str) or variant not in ("standard", "fast"):
        raise MaterialError("hacc_preset needs variant='standard' or variant='fast'")
    provider = _load_provider()
    _build_record(provider, None)
    hacc = _hacc_surface(provider)
    preset = hacc.helium_preset(ions, variant)
    if preset.variant != variant:
        raise BackendError(
            f"the provider returned the {preset.variant!r} haCC preset for the "
            f"{variant!r} variant"
        )
    source = f"pyroraptor@{_PROVIDER_REVISION}: {preset.source}"
    basis = HaccBasis(
        **cast(
            "dict[str, Any]",
            _hacc_fields(
                hacc,
                preset.atom,
                preset.basis,
                preset.gauge,
                preset.cut_energy,
                preset.max_step_au,
            ),
        ),
        source=source,
        preset=preset.identity,
    )
    if basis.variant != variant:
        raise BackendError(
            f"the provider's {variant!r} preset identity {preset.identity!r} does "
            "not name that variant"
        )
    return HaccPreset(
        basis=basis,
        variant=variant,
        step_margin=float(preset.step_margin),
        max_atomic_step=basis.max_step_au * AU_TIME,
        accuracy=float(preset.accuracy),
        box_sensitivity=float(preset.box_sensitivity),
        gauge_sensitivity=float(preset.gauge_sensitivity),
        max_peak_intensity=float(preset.max_peak_intensity),
        wavelength=float(preset.wavelength),
        window=float(preset.window),
        device_admissible=bool(preset.device_admissible),
        identity=preset.identity,
        source=source,
    )


def hacc_basis(
    atom: object,
    basis: object,
    gauge: object,
    cut_energy: float,
    max_step_au: float,
    *,
    source: str,
) -> HaccBasis:
    """Convert provider haCC objects into a custom, unqualified ``HaccBasis``.

    The "accept any provider model" path (SEP 20 section 5.1): the record's identity
    is derived from its full definition, and it is warned and recorded as unqualified
    when a run prepares it. Build its calibration record yourself, with
    ``status="research"``, no uncertainty and the limitation that the basis is
    unqualified.

    Parameters
    ----------
    atom : pyroraptor.hacc.Helium
        The atom and its ion count (1 or 2).
    basis : pyroraptor.hacc.Basis
        Radial axis in m, irECS and angular limit.
    gauge : pyroraptor.hacc.MixedGauge or pyroraptor.hacc.VelocityGauge
        The gauge; the velocity gauge becomes ``gauge_radius=None``.
    cut_energy : float
        Spectral cut in J.
    max_step_au : float
        RK4 step ceiling in atomic units of time.
    source : str
        Why this basis was chosen; recorded in its identity.

    Returns
    -------
    HaccBasis
        Custom record with ``preset=None``.

    Raises
    ------
    MaterialError
        For argon, an ion count other than 1 or 2, or any field the record cannot
        represent.
    BackendError
        If the optional provider is missing or has no haCC surface.
    """
    hacc = _hacc_surface(_load_provider())
    fields = _hacc_fields(hacc, atom, basis, gauge, cut_energy, max_step_au)
    return HaccBasis(**cast("dict[str, Any]", fields), source=source)


# ---------------------------------------------------------------- binding helpers


@dataclass(frozen=True)
class _Chunk:
    """Identify one stable contiguous radial batch.

    Attributes
    ----------
    start : int
        Inclusive nonnegative row index.
    stop : int
        Exclusive row index, strictly greater than start.
    """

    start: int
    stop: int

    @property
    def rows(self) -> int:
        """Return the number of rows in this contiguous chunk.

        Returns
        -------
        int
            Positive difference between the exclusive stop and inclusive start.
        """
        return self.stop - self.start


def _atomic_samples(descriptor: TDSEDescriptor) -> int:
    """Return the samples the provider integrates over: the window's or the lattice's.

    Parameters
    ----------
    descriptor : TDSEDescriptor
        Admitted descriptor.

    Returns
    -------
    int
        Retained atomic samples.
    """
    window = descriptor.atomic_window
    return descriptor.samples if window is None else window.samples


def _estimated_memory(
    descriptor: TDSEDescriptor,
    chunks: tuple[_Chunk, ...],
    native_outputs: str | None = None,
) -> dict[str, int]:
    """Estimate conservative data storage before allocating the native setup.

    Parameters
    ----------
    descriptor : TDSEDescriptor
        Admitted output contract and SI sampling lattice.
    chunks : tuple of _Chunk
        Nonempty stable partition of all radial rows.
    native_outputs : {'full', 'total_current'} or None, optional
        The provider's output selection when a full archive widens the
        descriptor's; None takes ``descriptor.outputs``.

    Returns
    -------
    dict
        Named conservative atomic storage estimates in bytes; vendor setup
        workspaces and allocator overhead are not hard bounded. The haCC provider
        offers no estimate before preparation (SEP 20, C7), so only its host
        histories are estimated; its model and plans are admitted from their
        measured ledgers.
    """
    samples = _atomic_samples(descriptor)
    outputs = descriptor.outputs if native_outputs is None else native_outputs
    histories = 48 if outputs == "full" else 20
    host = {
        "estimate.host_histories": 8 * histories * chunks[-1].stop * descriptor.samples
    }
    basis = descriptor.response.basis
    if not isinstance(basis, TDSEBasis):
        return host
    radial = sum(order - 1 for order in basis.radial_orders)
    radial += (basis.infinite_order or 0) - 1
    channels = basis.lmax + 1
    sizes = tuple(sorted({chunk.rows for chunk in chunks}))
    substeps = max(1, math.ceil(descriptor.spacing / descriptor.max_atomic_step))
    stages = 3 * (samples - 1) * substeps + 1
    return {
        "estimate.shared_dense_setup": 16 * 12 * channels * radial * radial,
        "estimate.plan_states": sum(
            16 * 24 * (descriptor.shadow_order + 1) * rows * channels * radial
            for rows in sizes
        ),
        "estimate.plan_stages": sum(8 * stages * (4 * rows + 16) for rows in sizes),
        **host,
    }


def _admit_memory(ledger: Mapping[str, int], limit: int | None) -> None:
    """Apply the declared combined optical and atomic byte budget.

    Parameters
    ----------
    ledger : mapping of str to int
        Named persistent allocations or estimates in bytes.
    limit : int or None
        Positive byte ceiling, or None for an unrestricted allocation.
    """
    total = sum(ledger.values())
    if limit is not None and total > limit:
        raise MemoryBudgetError(
            f"TDSE combined persistent-data admission requires {total} bytes, "
            f"exceeding {limit} bytes"
        )


def _json_default(value: object) -> object:
    """Convert a public native record or array into JSON-compatible containers.

    Parameters
    ----------
    value : object
        Dataclass, host array or NumPy scalar requiring conversion.

    Returns
    -------
    object
        Dictionary, list or Python scalar with no native resource handle.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported TDSE metadata value {type(value).__name__}")


def _encoded(value: object) -> str:
    """Encode finite metadata in a deterministic JSON representation.

    Parameters
    ----------
    value : object
        Nested public records, arrays and finite primitive values.

    Returns
    -------
    str
        Sorted JSON text; nonfinite values raise ValueError.
    """
    return json.dumps(value, default=_json_default, sort_keys=True, allow_nan=False)


def _failure_safe(value: object) -> object:
    """Preserve nonfinite failed monitors as explicit strings in valid JSON.

    Parameters
    ----------
    value : object
        Nested failure diagnostics, including possibly nonfinite scalars.

    Returns
    -------
    object
        Independent JSON-compatible containers with tagged nonfinite strings.
    """
    encoded = json.dumps(value, default=_json_default, sort_keys=True)
    return json.loads(encoded, parse_constant=lambda token: f"nonfinite:{token}")


def _input_diagnostics(
    fields: Float[np.ndarray, "n_nodes n_to"], spacing: float
) -> object:
    """Measure field endpoints and area without changing the supplied waveform.

    Parameters
    ----------
    fields : numpy.ndarray
        Finite SI fields in V/m, shape (n_nodes, n_to).
    spacing : float
        Positive sample spacing in s.

    Returns
    -------
    dict
        Per-node peak and endpoint fields in V/m and pulse area in V s/m.
    """
    return {
        "peak_field_v_per_m": np.max(np.abs(fields), axis=1),
        "leading_field_v_per_m": fields[:, 0].copy(),
        "trailing_field_v_per_m": fields[:, -1].copy(),
        "leading_eight_peak_v_per_m": np.max(np.abs(fields[:, :8]), axis=1),
        "trailing_eight_peak_v_per_m": np.max(np.abs(fields[:, -8:]), axis=1),
        "pulse_area_v_s_per_m": np.trapezoid(fields, dx=spacing, axis=1),
    }


def _drive_diagnostics(
    drive: Float[np.ndarray, "n_nodes n_window"], spacing: float
) -> object:
    """Record the tapered drive's area and vector-potential endpoint (SEP 20, 4.6).

    Multiplying by the gate need not preserve a zero-area drive; the area is
    recorded, never removed.

    Parameters
    ----------
    drive : numpy.ndarray
        The exact gated field passed to the provider, V/m.
    spacing : float
        Sample spacing in s.

    Returns
    -------
    dict
        Per-node area in V s/m, which is minus the vector potential at ``t_e``
        when ``A(t_s) = 0``, and the endpoint fields in V/m.
    """
    area = np.trapezoid(drive, dx=spacing, axis=1)
    return {
        "drive_area_v_s_per_m": area,
        "drive_final_vector_potential_v_s_per_m": -area,
        "drive_leading_field_v_per_m": drive[:, 0].copy(),
        "drive_trailing_field_v_per_m": drive[:, -1].copy(),
    }


def _row_chunks(execution: TDSEExecution, n_nodes: int) -> tuple[_Chunk, ...]:
    """Partition radial rows into stable contiguous batches.

    Parameters
    ----------
    execution : TDSEExecution
        Positive maximum batch size and device-independent resource limits.
    n_nodes : int
        Positive total number of radial rows.

    Returns
    -------
    tuple of _Chunk
        Ordered chunks covering each row exactly once.
    """
    if type(n_nodes) is not int or n_nodes < 1:
        raise BackendError("TDSE binding requires a positive integer n_nodes")
    return tuple(
        _Chunk(start, min(start + execution.max_batch_rows, n_nodes))
        for start in range(0, n_nodes, execution.max_batch_rows)
    )


def _classify(error: BaseException, provider: _Provider) -> str:
    """Classify a provider-side exception for the failure record (SEP 20, 5.6).

    Parameters
    ----------
    error : BaseException
        The exception raised by, or after, a provider call.
    provider : _Provider
        Loaded provider, for its ``DrivenError``.

    Returns
    -------
    str
        ``native`` for a provider ``DrivenError``, ``interface`` for a
        ``ValueError`` or ``TypeError`` from the provider, ``contract`` for a
        solprop validation of the returned record, ``unexpected`` otherwise.
    """
    if isinstance(error, provider.DrivenError):
        return "native"
    if isinstance(error, SolpropError):
        return "contract"
    if isinstance(error, ValueError | TypeError):
        return "interface"
    return "unexpected"


def _peak_intensity(fields: Float[np.ndarray, "n_nodes n_samples"]) -> list[float]:
    """Return each row's peak cycle-averaged intensity ``eps0 c E_max^2 / 2``.

    Parameters
    ----------
    fields : numpy.ndarray
        Real fields in V/m.

    Returns
    -------
    list of float
        Peak intensities in W/m^2, one per row.
    """
    peak = np.max(np.abs(fields), axis=1)
    return [float(value) for value in 0.5 * EPS0 * C * peak**2]


def estimate_binding_memory(
    descriptor: TDSEDescriptor,
    execution: TDSEExecution,
    n_nodes: int,
    *,
    archive: ArchiveOutputs | None = None,
) -> Mapping[str, int]:
    """Estimate atomic storage without importing the optional provider.

    Parameters
    ----------
    descriptor : TDSEDescriptor
        Resolved basis, fixed time lattice and requested response products.
    execution : TDSEExecution
        Stable maximum chunk size, determining the persistent plan shapes.
    n_nodes : int
        Positive total number of radial rows.
    archive : {'full', 'required'} or None, optional
        The output selection of a raw archive, whose writer buffers are added and
        whose full selection widens the provider histories; None without one.

    Returns
    -------
    mapping of str to int
        Conservative atomic host and device data estimates in bytes. Optical
        host storage is excluded and must be added for a shared run budget.
        Vendor allocation overhead and setup workspaces are not hard bounded.
        A haCC model has no estimate before preparation; its measured ledgers are
        admitted after preparation.

    Raises
    ------
    BackendError
        If the total row count is invalid.
    """
    chunks = _row_chunks(execution, n_nodes)
    estimate = _estimated_memory(
        descriptor, chunks, "full" if archive == "full" else None
    )
    if archive is not None:
        estimate.update(writer_memory(rows=chunks[0].rows, samples=descriptor.samples))
    return estimate


def prepare_binding(
    descriptor: TDSEDescriptor,
    execution: TDSEExecution,
    n_nodes: int,
    *,
    host_memory_bytes: int = 0,
    defer_plans: bool = False,
    archive: TDSEArchiveWriter | None = None,
) -> TDSEBinding:
    """Prepare persistent, stably chunked atomic plans for one response lattice.

    Parameters
    ----------
    descriptor : TDSEDescriptor
        Resolved scientific description and fixed fine time lattice in SI.
    execution : TDSEExecution
        Device, memory limit, stable maximum row count and atomic-call limit.
    n_nodes : int
        Positive total number of radial rows; fixed for the binding's lifetime.
    host_memory_bytes : int, optional
        Nonnegative optical host-storage estimate in bytes. Admission combines
        it with atomic host/device storage under ``execution.memory_limit``.
        It is not repeated in the binding's own memory ledger.
    defer_plans : bool, optional
        For a haCC model, prepare and admit the model only, and build the
        propagation plans when ``build_plans()`` is called. The run uses this so the
        consistency check's temporary one-row plan is built and closed before the
        propagation plans exist, and peak memory is the larger of the two, not their
        sum. Ignored for SAE, whose plans are always built here.
    archive : TDSEArchiveWriter or None, optional
        An open archive session (SEP 20 section 5.8). Every provider evaluation is
        written to it, input first and raw result before validation; a full
        archive builds the plans with the provider's full record. The caller owns
        and closes it. Its buffers are admitted with the binding's memory.

    Returns
    -------
    TDSEBinding
        Owning runtime adapter. Close it when the propagation ends. Evaluation
        requires a successful reservation of all logical calls in the trial.

    Raises
    ------
    BackendError
        If the provider, device, scientific identity or prepared state is invalid.
    MemoryBudgetError
        If estimated or measured persistent data exceed the declared byte limit.
    """
    if type(n_nodes) is not int or n_nodes < 1:
        raise BackendError("TDSE binding requires a positive integer n_nodes")
    if type(host_memory_bytes) is not int or host_memory_bytes < 0:
        raise BackendError("TDSE host_memory_bytes must be a nonnegative integer")
    if type(defer_plans) is not bool:
        raise BackendError("TDSE defer_plans must be a bool")
    if archive is not None and not isinstance(archive, TDSEArchiveWriter):
        raise BackendError("TDSE archive must be an open TDSEArchiveWriter")
    return _ProviderBinding(
        descriptor,
        execution,
        n_nodes,
        host_memory_bytes=host_memory_bytes,
        defer_plans=defer_plans,
        archive=archive,
    )


# ------------------------------------------------------------------ haCC model


class _HaccModel:
    """Prepare one haCC model, verify it and build its response plans.

    The model is entered into the caller's stack first, so it closes after every
    plan built on it (SEP 20, C1). A preset is checked against the installed
    provider's identity and fields; a custom basis is admitted as unqualified.

    Parameters
    ----------
    provider : _Provider
        Loaded, pin-checked provider.
    basis : HaccBasis
        Validated record.
    execution : TDSEExecution
        Device and memory limit.
    stack : ExitStack
        Owner of the model's lifetime.
    archive : {'required', 'full'}, optional
        ``"full"`` builds every plan with the provider's full record for a full
        archive; the source still reads the same total and order-one currents.
    """

    def __init__(
        self,
        provider: _Provider,
        basis: HaccBasis,
        execution: TDSEExecution,
        stack: ExitStack,
        *,
        archive: ArchiveOutputs = "required",
    ) -> None:
        if execution.complement == "vendor_blas":
            raise BackendError(
                "haCC response plans have no complement option; use the default "
                "TDSEExecution(complement='deterministic'), recorded as not "
                "applicable"
            )
        self.hacc = _hacc_surface(provider)
        self.basis = basis
        self.atom, self.native_basis, self.gauge = _hacc_objects(
            provider, self.hacc, basis
        )
        self.preset: _Native = None
        if basis.preset is not None:
            self.preset = self._verified_preset(execution)
            prepared = self.preset.prepare()
        else:
            prepared = self.hacc.prepare(
                self.atom,
                self.native_basis,
                gauge=self.gauge,
                cut_energy=basis.cut_energy,
            )
        self.model = stack.enter_context(prepared)
        self.memory = {
            f"model.{entry.name}": entry.bytes
            for entry in self.model.memory_ledger().entries
        }
        # The native output mode: compact with the escaped-electron closure,
        # full without it (the provider refuses compact order one then, C3), and
        # full throughout for a full archive. Whether the plans carry the closure
        # is known beforehand for a preset (always) and for the velocity gauge
        # (never); a custom mixed gauge learns it from its first plan.
        self.outputs: Literal["full", "total_current"] | None = (
            "total_current" if basis.preset is not None else None
        )
        self._closure: bool | None = True if basis.preset is not None else None
        if basis.gauge_radius is None:
            self.outputs = "full"
            self._closure = False
        if archive == "full":
            self.outputs = "full"

    def _verified_preset(self, execution: TDSEExecution) -> _Native:
        """Refuse a preset the installed provider does not reproduce.

        Parameters
        ----------
        execution : TDSEExecution
            Requested device family.

        Returns
        -------
        object
            The provider's ``HeliumPreset``.
        """
        preset = self.hacc.helium_preset(self.basis.ions, self.basis.variant)
        if preset.identity != self.basis.preset:
            raise BackendError(
                f"the installed provider's haCC preset is {preset.identity!r}, not "
                f"the expected {self.basis.preset!r}; regenerate the record"
            )
        fields = _hacc_fields(
            self.hacc,
            preset.atom,
            preset.basis,
            preset.gauge,
            preset.cut_energy,
            preset.max_step_au,
        )
        if fields != _numeric_fields(self.basis):
            raise BackendError(
                f"the installed provider's preset {preset.identity!r} does not "
                "reproduce the record's fields"
            )
        if execution.device != "cpu" and not preset.device_admissible:
            raise BackendError(
                f"the haCC preset {preset.identity!r} is not admitted on the "
                f"{execution.device} device route"
            )
        return preset

    @property
    def closure(self) -> bool:
        """Return whether the plans carry the escaped-electron closure.

        Returns
        -------
        bool
            True for a preset and for a custom mixed-gauge basis whose plans the
            provider qualified; False before a custom basis's first plan.
        """
        return self._closure is True

    def plan(self, *, rows: int, spacing: float, samples: int) -> _Native:
        """Construct one response plan, choosing the native output mode once.

        A custom mixed-gauge basis is first asked for the compact closed current;
        the provider refuses it with ``ValueError`` when the gauge radius admits no
        closure, and the full record is then used, as for the velocity gauge. For
        a full archive every plan is full, and a custom mixed-gauge basis takes its
        closure from the first plan's ``total_current_qualified``.

        Parameters
        ----------
        rows : int
            Atoms per evaluation.
        spacing : float
            Response spacing in s.
        samples : int
            Retained atomic samples.

        Returns
        -------
        object
            An unentered ``hacc.ResponsePlan``.
        """
        arguments = {
            "rows": rows,
            "spacing": float(spacing),
            "samples": samples,
            "max_step_au": float(self.basis.max_step_au),
            "field": "electric_field",
            "shadow_order": 1,
            "norm_growth_limit": HACC_NORM_GROWTH_LIMIT,
        }
        if self.outputs is None:
            try:
                plan = self.hacc.ResponsePlan(
                    self.model, outputs="total_current", **arguments
                )
            except ValueError as error:
                if "closure" not in str(error):
                    raise
                self.outputs = "full"
                self._closure = False
            else:
                self.outputs = "total_current"
                self._closure = True
                return plan
        plan = self.hacc.ResponsePlan(self.model, outputs=self.outputs, **arguments)
        if self._closure is None:
            self._closure = bool(plan.info.total_current_qualified)
        return plan

    def check_info(
        self, info: _Native, *, rows: int, samples: int, spacing: float
    ) -> None:
        """Require a plan or result's ``info`` to describe exactly this model.

        Parameters
        ----------
        info : object
            ``hacc.ResponsePlanInfo``.
        rows, samples : int
            Expected extents.
        spacing : float
            Expected spacing in s.
        """
        if (
            info.rows != rows
            or info.samples != samples
            or info.spacing != spacing
            or info.shadow_order != 1
            or info.outputs != self.outputs
            or info.atom != self.atom
            or info.basis != self.native_basis
            or info.gauge != self.gauge
            or info.cut_energy != self.basis.cut_energy
        ):
            raise BackendError(
                "haCC plan metadata disagree with the prepared model or contract"
            )
        if bool(info.total_current_qualified) != self.closure:
            if self.preset is not None:
                raise BackendError(
                    "provider contract violation: a haCC preset reports no "
                    "escaped-electron closure"
                )
            raise BackendError("haCC plan closure flag disagrees with its output mode")

    def record(self) -> dict[str, object]:
        """Return the model's provenance for run metadata.

        Returns
        -------
        dict
            Label, preset identity or custom flag, qualification, closure, the
            preset's accuracy and domain, and the complement policy.
        """
        preset = self.preset
        return {
            "label": self.basis.label,
            "basis_identity": self.basis.identity,
            "preset": self.basis.preset,
            "custom": self.basis.preset is None,
            "qualified": self.basis.qualified,
            "closure": self.closure,
            "native_outputs": self.outputs,
            "complement": "not_applicable",
            "preset_record": None
            if preset is None
            else {
                "identity": preset.identity,
                "variant": preset.variant,
                "step_margin": preset.step_margin,
                "accuracy": preset.accuracy,
                "box_sensitivity": preset.box_sensitivity,
                "gauge_sensitivity": preset.gauge_sensitivity,
                "max_peak_intensity": preset.max_peak_intensity,
                "wavelength": preset.wavelength,
                "window": preset.window,
                "device_admissible": preset.device_admissible,
            },
        }


# --------------------------------------------------------------------- binding


class _ProviderBinding:
    """Keep the optional runtime private to the driver and expose only the seam.

    Parameters
    ----------
    descriptor : TDSEDescriptor
        Admitted scientific configuration and fixed time lattice.
    execution : TDSEExecution
        Device family, persistent memory budget and attempted-call limit.
    n_nodes : int
        Positive total number of radial rows.
    host_memory_bytes : int, optional
        Nonnegative separately owned optical storage in bytes.
    defer_plans : bool, optional
        For haCC, build the propagation plans only in :meth:`build_plans`.
    archive : TDSEArchiveWriter or None, optional
        Open archive session receiving every evaluation.
    """

    def __init__(
        self,
        descriptor: TDSEDescriptor,
        execution: TDSEExecution,
        n_nodes: int,
        *,
        host_memory_bytes: int = 0,
        defer_plans: bool = False,
        archive: TDSEArchiveWriter | None = None,
    ) -> None:
        self._archive = archive
        # A full archive widens the provider's selection, never the descriptor's,
        # which keeps naming the products the source policy needs (SEP 20, 6.2).
        self._native_outputs: str = (
            "full"
            if archive is not None and archive.outputs == "full"
            else descriptor.outputs
        )
        self._trial: int | None = None
        self._trials_opened = 0
        self._trial_evaluations: list[int] = []
        self._defer_plans = defer_plans
        self._descriptor = descriptor
        self._execution = execution
        self._n_nodes = n_nodes
        self._chunks = _row_chunks(execution, n_nodes)
        self._host_memory_bytes = host_memory_bytes
        self._budget = AtomicCallBudget(execution.max_atomic_calls, len(self._chunks))
        self._stack = ExitStack()
        self._plans: dict[int, _Native] = {}
        self._plan_records: dict[str, object] = {}
        self._closed = False
        self._cached: TDSEResult | None = None
        self._cached_fields: Float[np.ndarray, "n_nodes n_to"] | None = None
        self._diagnostic_cache_hits = 0
        self._last_call: dict[str, object] = {}
        self._failures: list[object] = []
        self._hacc: _HaccModel | None = None
        self._window: AtomicWindowRecord | None = descriptor.atomic_window
        self._gate = descriptor.source_gate()
        self._samples = _atomic_samples(descriptor)
        self._domain_reasons: dict[str, object] = {}
        self._domain_flags: list[object] = []
        self._consistency: object = None
        self._estimate = _estimated_memory(
            descriptor, self._chunks, self._native_outputs
        )
        _admit_memory(
            {**self._estimate, "optical.host": host_memory_bytes},
            execution.memory_limit,
        )
        self._memory: dict[str, int] = (
            {}
            if archive is None
            else archive.memory_ledger(
                rows=self._chunks[0].rows, samples=descriptor.samples
            )
        )
        self._provider = _load_provider()
        self._build = _build_record(self._provider, execution)
        try:
            self._prepare()
            _admit_memory(
                {**self._memory, "optical.host": host_memory_bytes},
                execution.memory_limit,
            )
        except BaseException as error:
            self._stack.close()
            if not isinstance(error, Exception) or isinstance(error, SolpropError):
                raise
            kind = _classify(error, self._provider)
            raise BackendError(f"TDSE preparation failed: {error} [{kind}]") from error

    @property
    def native_outputs(self) -> str | None:
        """Return the provider's output selection for this binding's plans.

        Returns
        -------
        {'full', 'total_current'} or None
            The descriptor's selection, or ``"full"`` for a full archive; None for a
            custom haCC basis before its first plan decides it.
        """
        if self._hacc is not None:
            return self._hacc.outputs
        return self._native_outputs

    # -- preparation ---------------------------------------------------------------

    def _prepare(self) -> None:
        """Dispatch on the descriptor's model and admit the measured memory."""
        basis = self._descriptor.response.basis
        if isinstance(basis, HaccBasis):
            self._prepare_hacc(basis)
        else:
            self._prepare_sae(basis)
        # Include the simultaneously live provider histories, normalized chunks,
        # concatenated result and exact-input diagnostic cache in host admission.
        self._memory["host.result_and_cache_reserve"] = self._estimate[
            "estimate.host_histories"
        ]

    def _prepare_sae(self, basis: TDSEBasis) -> None:
        """Allocate shared SAE resources and one plan per distinct batch size.

        Parameters
        ----------
        basis : TDSEBasis
            Resolved SI atomic basis admitted against the estimate.
        """
        provider = self._provider
        model = provider.atomic_model(basis.potential)
        if model.version != basis.potential_version:
            raise BackendError(
                "TDSE atomic potential revision differs from its descriptor"
            )
        scaling = None
        if basis.scaling_radius is not None:
            assert basis.scaling_angle is not None and basis.infinite_order is not None
            assert basis.decay is not None
            scaling = provider.ComplexScaling(
                basis.scaling_radius,
                basis.scaling_angle,
                basis.infinite_order,
                basis.decay,
            )
        discretization = provider.Discretization(
            provider.RadialBasis(basis.radial_edges, basis.radial_orders),
            provider.AngularBasis(basis.lmax, basis.m),
            model,
            scaling,
            basis.cuts,
        )
        operators = self._stack.enter_context(discretization.assemble())
        descriptor = self._descriptor
        initial = provider.StateSelection(
            descriptor.initial_l, descriptor.initial_index
        )
        setup = self._stack.enter_context(
            provider.SpectralSetup(
                operators,
                provider.SpectralOptions(
                    cut_energy=basis.cut_energy,
                    states=(initial,),
                    growing=cast("Literal['remove']", descriptor.growing),
                    growth_tolerance=descriptor.growth_tolerance,
                    window_duration=descriptor.window_duration,
                    accept_removed_component=descriptor.accept_removed_component,
                ),
            )
        )
        for label, ledger in (
            ("operators", operators.memory_ledger()),
            ("setup", setup.memory_ledger()),
        ):
            for entry in ledger.entries:
                self._memory[f"{label}.{entry.name}"] = entry.bytes
        for rows in sorted({chunk.rows for chunk in self._chunks}):
            plan = self._stack.enter_context(
                provider.ResponsePlan(
                    setup,
                    rows=rows,
                    spacing=descriptor.spacing,
                    samples=self._samples,
                    max_step_au=(
                        descriptor.max_atomic_step
                        / provider.unit_scale("time").si_per_atomic_unit
                    ),
                    initial=initial,
                    field="electric_field",
                    shadow_order=descriptor.shadow_order,
                    complement=self._execution.complement,
                    outputs=self._native_outputs,
                )
            )
            self._plans[rows] = plan
            for entry in plan.memory_ledger().entries:
                if not entry.shared:
                    self._memory[f"plan.{rows}.{entry.name}"] = entry.bytes
        self._plan_records = {
            str(rows): asdict(plan.info) for rows, plan in self._plans.items()
        }

    def _prepare_hacc(self, basis: HaccBasis) -> None:
        """Prepare the haCC model, admit its ledger and build the chunk plans.

        Parameters
        ----------
        basis : HaccBasis
            Validated preset or custom record.
        """
        model = _HaccModel(
            self._provider,
            basis,
            self._execution,
            self._stack,
            archive="required" if self._archive is None else self._archive.outputs,
        )
        self._hacc = model
        self._memory.update(model.memory)
        # Admit the model's measured ledger before any plan is built (C7).
        _admit_memory(
            {**self._memory, "optical.host": self._host_memory_bytes},
            self._execution.memory_limit,
        )
        if basis.preset is not None:
            self._preset_domain_at_start(model)
        if not self._defer_plans:
            self._build_hacc_plans()

    def _build_hacc_plans(self) -> None:
        """Build one haCC plan per chunk size, then warn for a custom basis.

        The custom-basis warning waits for the plans because whether the model
        has the escaped-electron closure is known only once a plan exists.
        """
        model = self._hacc
        assert model is not None
        basis = model.basis
        spacing = float(self._descriptor.spacing)
        for rows in sorted({chunk.rows for chunk in self._chunks}):
            plan = self._stack.enter_context(
                model.plan(rows=rows, spacing=spacing, samples=self._samples)
            )
            model.check_info(
                plan.info, rows=rows, samples=self._samples, spacing=spacing
            )
            self._plans[rows] = plan
            for entry in plan.memory_ledger().entries:
                if not entry.shared:
                    self._memory[f"plan.{rows}.{entry.name}"] = entry.bytes
        self._plan_records = {
            str(rows): asdict(plan.info) for rows, plan in self._plans.items()
        }
        if basis.preset is None:
            missing = (
                ""
                if model.closure
                else "; its total current lacks the escaped-electron closure, so the "
                "source uses the in-box first order and a total that omits the "
                "momentum carried out through R0"
            )
            warnings.warn(
                f"{basis.label} is a custom haCC basis with no qualified accuracy "
                f"and no preset domain; it is recorded as unqualified{missing}",
                TDSEUnqualifiedModelWarning,
                stacklevel=5,
            )

    @property
    def plans_built(self) -> bool:
        """Return whether the propagation plans exist.

        Returns
        -------
        bool
            False only for a haCC binding prepared with ``defer_plans`` before
            :meth:`build_plans`.
        """
        return self._hacc is None or bool(self._plans)

    def build_plans(self) -> None:
        """Build and admit a deferred haCC binding's propagation plans.

        Call once, after any consistency diagnostic, whose temporary plan is then
        already closed. A failure closes every resource, as at preparation.

        Raises
        ------
        BackendError
            If the binding is closed, has no deferred plans, or a plan fails.
        MemoryBudgetError
            If the plans do not fit beside the model under the memory limit.
        """
        self._ensure_open()
        if self.plans_built:
            raise BackendError("the TDSE binding's plans are already built")
        try:
            self._build_hacc_plans()
            _admit_memory(
                {**self._memory, "optical.host": self._host_memory_bytes},
                self._execution.memory_limit,
            )
        except BaseException as error:
            self.close()
            if not isinstance(error, Exception) or isinstance(error, SolpropError):
                raise
            kind = _classify(error, self._provider)
            raise BackendError(f"TDSE preparation failed: {error} [{kind}]") from error

    def _preset_domain_at_start(self, model: _HaccModel) -> None:
        """Flag a preset run's duration or carrier beyond its qualification.

        Parameters
        ----------
        model : _HaccModel
            Prepared preset model.
        """
        preset = model.preset
        configuration = json.loads(self._descriptor.configuration_json)
        lattice = configuration["lattice"]
        duration = lattice["atomic_duration"]
        if duration > preset.window:
            self._flag_domain(
                "duration",
                {"atomic_duration": duration, "preset_window": preset.window},
                f"the atomic duration {duration:.6g} s exceeds the preset's tested "
                f"window {preset.window:.6g} s",
            )
        carrier = lattice["reference_wavelength"]
        if not math.isclose(carrier, preset.wavelength, rel_tol=1e-12):
            self._flag_domain(
                "carrier",
                {
                    "reference_wavelength": carrier,
                    "preset_wavelength": preset.wavelength,
                },
                f"the reference wavelength {carrier:.6g} m differs from the preset's "
                f"qualifying carrier {preset.wavelength:.6g} m",
            )

    def _flag_domain(self, reason: str, record: object, message: str) -> None:
        """Record a domain reason and warn once per reason per run.

        Parameters
        ----------
        reason : {'duration', 'carrier', 'intensity'}
            Why the run is outside the preset's domain.
        record : object
            JSON-compatible facts.
        message : str
            Human-readable warning text.
        """
        if reason in self._domain_reasons:
            return
        self._domain_reasons[reason] = record
        assert self._hacc is not None
        warnings.warn(
            f"{self._hacc.basis.label} preset outside its qualified domain: {message}; "
            "the run proceeds and the record is flagged",
            TDSEDomainWarning,
            stacklevel=4,
        )

    # -- accounting ----------------------------------------------------------------

    @property
    def ledger(self) -> AtomicCallLedger:
        """Return attempted-call accounting, including failed evaluations.

        Returns
        -------
        AtomicCallLedger
            Immutable snapshot of charged, reserved and remaining call counts.
        """
        return self._budget.snapshot()

    def reserve(self, logical_calls: int) -> None:
        """Reserve every chunk of a complete trial before its first atomic call.

        A successful reservation opens the next optical trial, whose evaluations
        the archive labels with its identifier (SEP 20 section 5.8).

        Parameters
        ----------
        logical_calls : int
            Positive number of complete all-node evaluations required by the trial.
        """
        self._ensure_open()
        self._ensure_plans()
        self._budget.reserve(logical_calls)
        self._trial = self._trials_opened
        self._trials_opened += 1
        self._trial_evaluations = []

    def record_trial(self, *, accepted: bool) -> None:
        """Record the stepper's decision on the latest trial in the archive.

        Parameters
        ----------
        accepted : bool
            Whether the trial's step was accepted.

        Raises
        ------
        BackendError
            If no trial was reserved.
        TDSEArchiveError
            If the archive cannot record it; the run then stops before its next
            provider call.
        """
        if self._trial is None:
            raise BackendError("a TDSE trial outcome needs a reserved trial")
        if self._archive is not None:
            self._archive.record_trial(
                self._trial, accepted=accepted, evaluations=self._trial_evaluations
            )

    def release_reservation(self) -> None:
        """Release unused capacity while retaining every attempted-call charge."""
        self._budget.release_reservation()

    def restore_ledger(self, ledger: AtomicCallLedger) -> None:
        """Restore prior attempts before performing work in a resumed binding.

        Parameters
        ----------
        ledger : AtomicCallLedger
            Previous checkpoint accounting, preserving charges with the new limit.
        """
        self._ensure_open()
        self._budget.restore(ledger)

    def _ensure_open(self) -> None:
        """Reject use after native resources have been closed."""
        if self._closed:
            raise BackendError("TDSE binding is closed")

    def _ensure_plans(self) -> None:
        """Reject propagation before a deferred binding has built its plans."""
        if not self.plans_built:
            raise BackendError(
                "the TDSE binding's propagation plans are not built; call build_plans"
            )

    # -- evaluation ----------------------------------------------------------------

    def _drive(
        self, fields: Float[np.ndarray, "n_nodes n_to"], chunk: _Chunk
    ) -> Float[np.ndarray, "n_rows n_samples"]:
        """Return the exact float64 field passed to the provider for one chunk.

        Parameters
        ----------
        fields : numpy.ndarray
            Full-lattice nodal fields in V/m.
        chunk : _Chunk
            Rows of this call.

        Returns
        -------
        numpy.ndarray
            Contiguous rows, sliced to the window and multiplied by its gate.
        """
        rows = fields[chunk.start : chunk.stop]
        window = self._window
        if window is None:
            return rows
        assert self._gate is not None
        span = slice(window.first, window.last + 1)
        return np.ascontiguousarray(rows[:, span] * self._gate[span])

    def _pad(
        self, values: Float[np.ndarray, "n_rows n_samples"]
    ) -> Float[np.ndarray, "n_rows n_to"]:
        """Place a window's currents on the full lattice with zeros outside.

        Parameters
        ----------
        values : numpy.ndarray
            Provider rows on the retained samples.

        Returns
        -------
        numpy.ndarray
            Rows on the full lattice; unchanged without a window.
        """
        window = self._window
        if window is None:
            return values
        full = np.zeros((values.shape[0], self._descriptor.samples), dtype=values.dtype)
        full[:, window.first : window.last + 1] = values
        return full

    def _pad_valid(self, valid: np.ndarray) -> np.ndarray:
        """Place a window's validity on the full lattice; undriven samples are valid.

        Parameters
        ----------
        valid : numpy.ndarray
            Boolean provider validity on the retained samples.

        Returns
        -------
        numpy.ndarray
            Boolean validity on the full lattice.
        """
        window = self._window
        if window is None:
            return valid
        # The provider marks a row invalid at and after its first failure, and the
        # padded samples after the window would then read as valid. That cannot hide
        # a failure: a failed outcome raises in _normalize and any invalid sample
        # inside the window raises in TDSEResult, so this array reaches the source
        # only when every driven sample is valid. Outside the window the atom is not
        # driven and its padded current is exactly zero.
        full = np.ones((valid.shape[0], self._descriptor.samples), dtype=bool)
        full[:, window.first : window.last + 1] = valid
        return full

    def evaluate(  # noqa: PLR0912, PLR0915 - one evaluation's complete sequence
        self,
        fields: Float[np.ndarray, "n_nodes n_to"],
        *,
        z: float,
        purpose: TDSEPurpose,
        density: float | None = None,
    ) -> TDSEResult:
        """Evaluate unchanged node fields from the prepared ground state.

        With an archive, the evaluation's input is written before its first
        provider call and each chunk's raw result as soon as it returns, before
        validation; a write failure stops the evaluation before the next provider
        call (SEP 20 section 5.8).

        Parameters
        ----------
        fields : numpy.ndarray
            Finite float64 fields in V/m, shape (n_nodes, n_to).
        z : float
            Finite optical position in m for provenance and failure reports.
        purpose : {'startup', 'predictor', 'corrected', 'diagnostic'}
            Reason for the source evaluation; only exact diagnostics may reuse data.
        density : float or None, optional
            Finite nonnegative number density at `z` in m^-3, an archive label.

        Returns
        -------
        TDSEResult
            Owning, read-only, unscaled microscopic SI currents and shadow dipoles.
        """
        self._ensure_open()
        self._ensure_plans()
        identity = waveform_identity(fields)
        if fields.shape != (self._n_nodes, self._descriptor.samples):
            raise BackendError(
                "TDSE fields changed their prepared row count or sample count"
            )
        if not math.isfinite(z) or purpose not in (
            "startup",
            "predictor",
            "corrected",
            "diagnostic",
        ):
            raise BackendError(
                "TDSE evaluation needs a finite z and a declared purpose"
            )
        if density is not None and (
            not isinstance(density, float | int)
            or isinstance(density, bool)
            or not math.isfinite(density)
            or density < 0
        ):
            raise BackendError("TDSE density label must be finite and nonnegative")
        if (
            purpose == "diagnostic"
            and self._cached is not None
            and identity == self._cached.input_identity
            and self._cached_fields is not None
            and np.array_equal(fields, self._cached_fields)
        ):
            self._diagnostic_cache_hits += 1
            return self._cached
        archive = self._archive
        if archive is not None:
            archive.ensure_writable()
        drives = [self._drive(fields, chunk) for chunk in self._chunks]
        domain = self._call_domain(fields, drives, z=z, purpose=purpose)
        evaluation = self._archive_begin(
            fields,
            drives,
            labels={
                "trial_id": self._trial,
                "role": _ARCHIVE_ROLES[purpose],
                "purpose": purpose,
                "z": float(z),
                "density": None if density is None else float(density),
                "input_identity": identity,
            },
        )
        try:
            self._budget.begin_evaluation(purpose=purpose)
        except BackendError as error:
            self._archive_failure(evaluation, None, error, "contract")
            raise
        self._last_call = {
            "input_identity": identity,
            "configuration_identity": self._descriptor.identity,
            "z": z,
            "purpose": purpose,
            "input_diagnostics": _input_diagnostics(fields, self._descriptor.spacing),
            "chunks": [],
        }
        if self._window is not None:
            self._last_call["drive_diagnostics"] = _drive_diagnostics(
                np.concatenate(drives, axis=0), self._descriptor.spacing
            )
        if domain is not None:
            self._last_call["domain"] = domain
        results: list[TDSEResult] = []
        chunk_records: list[object] = []
        self._last_call["chunks"] = chunk_records
        for index, (chunk, drive) in enumerate(zip(self._chunks, drives, strict=True)):
            # A failed write raises before this point, so no provider call follows
            # one (SEP 20 section 5.8); the writer refuses any later evaluation.
            self._budget.charge(purpose=purpose, z=z)
            raw: _Native = None
            archived: TDSEArchiveError | None = None
            try:
                # Public evaluate always resets every row to the prepared state.
                raw = self._plans[chunk.rows].evaluate(drive)
                # Captured as returned, before any validation (SEP 20, 5.8).
                archived = self._archive_result(evaluation, index, raw)
                self._validate_diagnostics(raw, chunk.rows, self._samples)
                chunk_records.append(self._native_record(raw, chunk))
                results.append(self._normalize(raw, chunk, identity))
            except Exception as error:
                self._budget.record_failure()
                classification = _classify(error, self._provider)
                failure = {
                    "z": z,
                    "row_start": chunk.start,
                    "row_stop": chunk.stop,
                    "input_identity": identity,
                    "classification": classification,
                    "error": f"{type(error).__name__}: {error}",
                    "native": None
                    if raw is None
                    else self._failed_native_record(raw, chunk),
                }
                self._failures.append(_failure_safe(failure))
                self._budget.release_reservation()
                wrapped = BackendError(
                    f"TDSE failure at z={z:.17g} m, "
                    f"rows [{chunk.start}, {chunk.stop}), "
                    f"input={identity}: {error} [{classification}]"
                )
                if archived is not None:
                    wrapped.add_note(
                        f"the TDSE archive could not store this result: {archived}"
                    )
                self._archive_failure(
                    evaluation, index, error, classification, note=wrapped
                )
                raise wrapped from error
            if archived is not None:
                self._budget.release_reservation()
                raise archived
        result = self._combine(results, identity)
        self._cached = result
        self._cached_fields = np.array(fields, dtype=np.float64, order="C", copy=True)
        self._cached_fields.setflags(write=False)
        if archive is not None and evaluation is not None:
            try:
                archive.finish_evaluation(evaluation, "returned")
            except TDSEArchiveError:
                self._budget.release_reservation()
                raise
        return result

    # -- archive -------------------------------------------------------------------

    def _archive_begin(
        self,
        fields: Float[np.ndarray, "n_nodes n_to"],
        drives: list[Float[np.ndarray, "_ n_samples"]],
        *,
        labels: Mapping[str, object],
    ) -> int | None:
        """Write an evaluation's labels and complete input before any provider call.

        Parameters
        ----------
        fields : numpy.ndarray
            The full-lattice nodal fields, V/m.
        drives : list of numpy.ndarray
            The exact provider input of each chunk, V/m.
        labels : mapping
            Role, trial, position, density and input identity.

        Returns
        -------
        int or None
            The archive's evaluation identifier, or None without an archive.
        """
        archive = self._archive
        if archive is None:
            return None
        descriptor = self._descriptor
        window = self._window
        origin = float(lattice_times(descriptor.samples, descriptor.spacing)[0])
        identifier = archive.begin_evaluation(
            {
                **labels,
                "configuration_identity": descriptor.identity,
                "basis_identity": descriptor.response.basis.identity,
                "model": descriptor.model,
                "descriptor_outputs": descriptor.outputs,
                "native_outputs": self.native_outputs,
                "archive_outputs": archive.outputs,
                "lattice": "optical",
                "lattice_samples": descriptor.samples,
                "spacing": float(descriptor.spacing),
                "optical_time_origin": origin,
                "window_first": 0 if window is None else window.first,
                "window_last": descriptor.samples - 1
                if window is None
                else window.last,
                "provider_samples": self._samples,
                "provider_time_zero": origin if window is None else window.t_s,
            },
            prewindow=fields,
            drives=[
                (chunk.start, chunk.stop, drive)
                for chunk, drive in zip(self._chunks, drives, strict=True)
            ],
            gate=None if window is None else self._gate,
            binding=self._binding_record(),
        )
        self._trial_evaluations.append(identifier)
        return identifier

    def _binding_record(self) -> dict[str, object]:
        """Return the build, execution, plan and chunk records an archive shares.

        Returns
        -------
        dict
            Provider plan/build and execution configuration.
        """
        return {
            "build": self._build,
            "execution": asdict(self._execution),
            "plans": self._plan_records,
            "chunks": [asdict(chunk) for chunk in self._chunks],
            "hacc": None if self._hacc is None else self._hacc.record(),
        }

    def _archive_result(
        self, evaluation: int | None, index: int, raw: _Native
    ) -> TDSEArchiveError | None:
        """Store one returned record, returning a failure instead of raising it.

        The binding's own validation runs next either way: a contract failure of
        the same result is then the primary error, and the archive failure a note.

        Parameters
        ----------
        evaluation : int or None
            The archive's evaluation, or None without an archive.
        index : int
            The chunk index.
        raw : object
            The provider's record, as returned.

        Returns
        -------
        TDSEArchiveError or None
            The write failure, if any.
        """
        archive = self._archive
        if archive is None or evaluation is None:
            return None
        outputs = self.native_outputs
        assert outputs is not None
        try:
            archive.write_result(
                evaluation,
                index,
                raw,
                record_serializer(self._descriptor.model, outputs),
            )
        except TDSEArchiveError as error:
            return error
        return None

    def _archive_failure(
        self,
        evaluation: int | None,
        index: int | None,
        error: BaseException,
        classification: str,
        *,
        note: BaseException | None = None,
    ) -> None:
        """Record an evaluation's exception, keeping the caller's error primary.

        Parameters
        ----------
        evaluation : int or None
            The archive's evaluation, or None without an archive.
        index : int or None
            The chunk that raised, or None.
        error : BaseException
            The exception.
        classification : str
            Its failure class.
        note : BaseException or None, optional
            The error the caller raises, which gains a note if recording fails.
        """
        archive = self._archive
        if archive is None or evaluation is None:
            return
        try:
            archive.write_failure(
                evaluation, chunk=index, error=error, classification=classification
            )
            archive.finish_evaluation(evaluation, "raised")
        except TDSEArchiveError as failure:
            (error if note is None else note).add_note(
                f"the TDSE archive could not record this failure: {failure}"
            )

    def _call_domain(
        self,
        fields: Float[np.ndarray, "n_nodes n_to"],
        drives: list[Float[np.ndarray, "_ n_samples"]],
        *,
        z: float,
        purpose: TDSEPurpose,
    ) -> dict[str, object] | None:
        """Flag a preset call whose peak input intensity exceeds the preset's bound.

        The peaks before and after the drive taper are both recorded, and the flag
        uses the larger, so the gate cannot conceal a field above the bound.

        Parameters
        ----------
        fields : numpy.ndarray
            Full-lattice nodal fields in V/m.
        drives : list of numpy.ndarray
            The gated provider inputs of each chunk.
        z : float
            Optical position in m.
        purpose : str
            Evaluation label.

        Returns
        -------
        dict or None
            The call's domain record for a preset, None otherwise.
        """
        model = self._hacc
        if model is None or model.preset is None:
            return None
        bound = float(model.preset.max_peak_intensity)
        before = _peak_intensity(fields)
        after = _peak_intensity(np.concatenate(drives, axis=0))
        outside = max(before) > bound
        record: dict[str, object] = {
            "peak_intensity_w_per_m2": before,
            "driven_peak_intensity_w_per_m2": after,
            "max_peak_intensity_w_per_m2": bound,
            "outside_preset_domain": outside,
        }
        if outside:
            self._domain_flags.append({"z": z, "purpose": purpose, **record})
            self._flag_domain(
                "intensity",
                {"max_peak_intensity_w_per_m2": bound},
                f"a call's peak input intensity {max(before):.6g} W/m^2 exceeds "
                f"{bound:.6g} W/m^2",
            )
        return record

    def _bound(self, raw: _Native) -> _Native:
        """Return a result's neglected-current bound, wherever its record keeps it.

        Parameters
        ----------
        raw : object
            Provider result.

        Returns
        -------
        object
            The bound array, or None for a haCC model without the closure.
        """
        if self._hacc is None and isinstance(raw, self._provider.GridResponse):
            return raw.plasma.neglected_current_bound
        return raw.neglected_current_bound

    def _failed_native_record(self, raw: _Native, chunk: _Chunk) -> object:
        """Retain diagnostic facts even when the native result is malformed.

        Parameters
        ----------
        raw : object
            Public provider result returned by the failed evaluation.
        chunk : _Chunk
            Contiguous row range to identify in the failure record.

        Returns
        -------
        object
            Failure-safe diagnostic containers, or the serialization failure reason.
        """
        try:
            return _failure_safe(
                {
                    "row_start": chunk.start,
                    "row_stop": chunk.stop,
                    "monitors": raw.monitors,
                    "outcomes": raw.outcomes,
                    "valid": raw.valid,
                    "neglected_current_bound": self._bound(raw),
                }
            )
        except (AttributeError, TypeError, ValueError) as error:
            return {"unreadable_native_record": type(raw).__name__, "error": str(error)}

    def _validate_diagnostics(self, raw: _Native, rows: int, samples: int) -> None:
        """Reject malformed diagnostics before reducing or labelling row arrays.

        Parameters
        ----------
        raw : object
            Public provider result with bounds and row monitors.
        rows : int
            Expected row count.
        samples : int
            Expected atomic sample count.
        """
        if self._hacc is not None and (
            not isinstance(raw.total_current, np.ndarray)
            or raw.total_current.shape != (rows, samples)
            or not np.all(np.isfinite(raw.total_current))
        ):
            raise BackendError("haCC total current is malformed or non-finite")
        bound = self._bound(raw)
        absent_admitted = self._hacc is not None and not self._hacc.closure
        if bound is None and absent_admitted:
            pass
        elif (
            not isinstance(bound, np.ndarray)
            or bound.dtype != np.dtype(np.float64)
            or bound.shape != (rows, samples)
            or not np.all(np.isfinite(bound))
            or np.any(bound < 0)
        ):
            raise BackendError("TDSE neglected-current bounds are malformed")
        # The haCC monitors carry a bool non_finite flag beside float64 norms and
        # the int64 step index (SEP 20, C4).
        kinds = {"monitored_step": np.int64, "non_finite": np.bool_}
        for name, vector in asdict(raw.monitors).items():
            if vector is None:
                continue
            dtype = np.dtype(kinds.get(name, np.float64))
            if (
                not isinstance(vector, np.ndarray)
                or vector.dtype != dtype
                or vector.shape != (rows,)
                or (dtype == np.float64 and not np.all(np.isfinite(vector)))
            ):
                raise BackendError(f"TDSE row monitor {name!r} is malformed")

    def _native_record(self, raw: _Native, chunk: _Chunk) -> object:
        """Summarise validated native diagnostics with their original row positions.

        Parameters
        ----------
        raw : object
            Public result whose diagnostic shapes and finiteness were validated.
        chunk : _Chunk
            Contiguous row range of this result.

        Returns
        -------
        dict
            Per-row monitors, outcomes, validity and maximum current bounds. A haCC
            record adds the bound's maximum relative to the row's peak total current,
            ``None`` with ``zero_reference`` for a zero-current row; the bound is a
            diagnostic and never compared with a threshold.
        """
        bound = self._bound(raw)
        record: dict[str, object] = {
            "row_start": chunk.start,
            "row_stop": chunk.stop,
            "monitors": raw.monitors,
            "outcomes": raw.outcomes,
            "valid": np.all(raw.valid, axis=1),
            "neglected_current_bound_max": None
            if bound is None
            else np.max(bound, axis=1),
        }
        if self._hacc is not None:
            peaks = np.max(np.abs(raw.total_current), axis=1)
            maxima = None if bound is None else np.max(bound, axis=1)
            record["neglected_current_bound"] = (
                None
                if maxima is None
                else [
                    {
                        "relative_to_peak": None if peak == 0 else float(top / peak),
                        "zero_reference": bool(peak == 0),
                    }
                    for top, peak in zip(maxima, peaks, strict=True)
                ]
            )
        return record

    def _hacc_currents(
        self, raw: _Native, rows: int, samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate a haCC result and return its total and first-order currents.

        Parameters
        ----------
        raw : object
            ``hacc.TotalCurrentResponse`` or ``hacc.GridResponse``.
        rows, samples : int
            Expected extents.

        Returns
        -------
        tuple of numpy.ndarray
            ``j_tot`` and ``j_1`` on the retained samples, C m/s per atom.
        """
        model = self._hacc
        assert model is not None
        model.check_info(
            raw.info,
            rows=rows,
            samples=samples,
            spacing=float(self._descriptor.spacing),
        )
        if model.outputs == "total_current":
            if not isinstance(raw, model.hacc.TotalCurrentResponse):
                raise BackendError(
                    "TDSE compact contract returned a different result type"
                )
            if raw.shadow_orders != (1,) or raw.shadow_total_current is None:
                raise BackendError(
                    "TDSE compact response has incorrect shadow-order labels"
                )
            if raw.shadow_total_current.shape != (rows, 1, samples):
                raise BackendError(
                    "TDSE compact response has incorrect shadow array extents"
                )
            return raw.total_current, raw.shadow_total_current[:, 0, :]
        if not isinstance(raw, model.hacc.GridResponse) or raw.shadow is None:
            raise BackendError("haCC full output requires a response with shadows")
        if raw.shadow.orders != (1,):
            raise BackendError("TDSE full response has incorrect shadow-order labels")
        # With the closure the total's order one is returned; without it the
        # in-box order one is the first order of the field-impulse total, whose
        # added term is at least third order (SEP 20, 4.3).
        first = raw.shadow.component("total_current" if model.closure else "current", 1)
        return raw.total_current, first

    def _sae_currents(
        self, raw: _Native, chunk: _Chunk
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, tuple[int, ...]
    ]:
        """Validate an SAE result and return its currents and shadow products.

        Parameters
        ----------
        raw : object
            SAE ``GridResponse`` or ``TotalCurrentResponse``.
        chunk : _Chunk
            Expected rows.

        Returns
        -------
        tuple
            Total and first-order currents, the optional third-order current and
            dipole, and the order labels, on the retained samples.
        """
        descriptor = self._descriptor
        expected_orders = (1,) if descriptor.shadow_order == 1 else (1, 3)
        info = raw.info
        if (
            info.rows != chunk.rows
            or info.samples != self._samples
            or info.spacing != descriptor.spacing
            or info.shadow_order != descriptor.shadow_order
            or info.outputs != self._native_outputs
        ):
            raise BackendError(
                "TDSE result metadata disagree with the prepared contract"
            )
        if self._native_outputs == "full":
            if not isinstance(raw, self._provider.GridResponse) or raw.shadow is None:
                raise BackendError(
                    "TDSE shadow model requires a full response with shadow products"
                )
            orders = raw.shadow.orders
            if orders != expected_orders:
                raise BackendError(
                    "TDSE full response has incorrect shadow-order labels"
                )
            # A full archive reads the same total and shadow currents from the full
            # record; only a descriptor that needs it takes the third-order dipole.
            return (
                raw.plasma.total_current,
                raw.shadow.component("total_current", 1),
                raw.shadow.component("total_current", 3) if 3 in orders else None,
                raw.shadow.component("charge_dipole", 3)
                if descriptor.outputs == "full"
                else None,
                orders,
            )
        if not isinstance(raw, self._provider.TotalCurrentResponse):
            raise BackendError("TDSE compact contract returned a different result type")
        orders = raw.shadow_orders
        if orders != expected_orders or raw.shadow_total_current is None:
            raise BackendError(
                "TDSE compact response has incorrect shadow-order labels"
            )
        if raw.shadow_total_current.shape != (chunk.rows, len(orders), self._samples):
            raise BackendError(
                "TDSE compact response has incorrect shadow array extents"
            )
        third = raw.shadow_total_current[:, orders.index(3), :] if 3 in orders else None
        return (
            raw.total_current,
            raw.shadow_total_current[:, orders.index(1), :],
            third,
            None,
            orders,
        )

    def _normalize(self, raw: _Native, chunk: _Chunk, identity: str) -> TDSEResult:
        """Validate native labels and copy one chunk into the host seam result.

        Parameters
        ----------
        raw : object
            Full or compact public provider result.
        chunk : _Chunk
            Expected row range with fixed sample count.
        identity : str
            Hash of the complete all-node input waveform.

        Returns
        -------
        TDSEResult
            Validated owning SI result without density or calibration factors,
            on the full lattice.
        """
        descriptor = self._descriptor
        if self._hacc is not None:
            total, first = self._hacc_currents(raw, chunk.rows, self._samples)
            third = dipole = None
            orders: tuple[int, ...] = (1,)
        else:
            total, first, third, dipole, orders = self._sae_currents(raw, chunk)
        if len(raw.outcomes) != chunk.rows or any(
            not outcome.ok for outcome in raw.outcomes
        ):
            raise BackendError("TDSE provider reported a failed atomic row")
        if total.shape != (chunk.rows, self._samples):
            raise BackendError("TDSE provider returned incorrect current extents")
        # A window admits tdse_native only (it requires the time collar), so no
        # third-order product is ever padded.
        return TDSEResult(
            self._pad(total),
            self._pad(first),
            input_identity=identity,
            configuration_identity=descriptor.identity,
            valid=self._pad_valid(raw.valid),
            shadow_orders=orders,
            third_order_current=third,
            third_order_dipole=dipole,
            monitors_json=_encoded(self._native_record(raw, chunk)),
            outcomes_json=_encoded(
                [
                    {"row": chunk.start + index, **asdict(value)}
                    for index, value in enumerate(raw.outcomes)
                ]
            ),
        )

    def _combine(self, results: list[TDSEResult], identity: str) -> TDSEResult:
        """Concatenate normalized chunks in the original radial row order.

        Parameters
        ----------
        results : list of TDSEResult
            Nonempty ordered chunk results with identical scientific contracts.
        identity : str
            Hash of the complete all-node input waveform.

        Returns
        -------
        TDSEResult
            Owning all-node result with merged native diagnostics.
        """
        third = None
        dipole = None
        if self._descriptor.shadow_order == 3:
            currents = [result.third_order_current for result in results]
            assert all(value is not None for value in currents)
            third = np.concatenate(
                [value for value in currents if value is not None], axis=0
            )
        if self._descriptor.outputs == "full":
            dipoles = [result.third_order_dipole for result in results]
            assert all(value is not None for value in dipoles)
            dipole = np.concatenate(
                [value for value in dipoles if value is not None], axis=0
            )
        return TDSEResult(
            np.concatenate([result.total_current for result in results], axis=0),
            np.concatenate([result.first_order_current for result in results], axis=0),
            input_identity=identity,
            configuration_identity=self._descriptor.identity,
            valid=np.concatenate([result.valid for result in results], axis=0),
            shadow_orders=results[0].shadow_orders,
            third_order_current=third,
            third_order_dipole=dipole,
            monitors_json=_encoded(
                {"chunks": [json.loads(result.monitors_json) for result in results]}
            ),
            outcomes_json=_encoded(
                [
                    outcome
                    for result in results
                    for outcome in json.loads(result.outcomes_json)
                ]
            ),
        )

    # -- the haCC consistency diagnostic --------------------------------------------

    def diagnostic_currents(
        self, field: Float[np.ndarray, "1 n_samples"], spacing: float
    ) -> tuple[Float[np.ndarray, " n_samples"], Float[np.ndarray, " n_samples"]]:
        """Evaluate one diagnostic row on the run's open haCC model (SEP 20, 4.9).

        A temporary one-row plan on the protocol lattice is built, admitted with the
        propagation plans against the memory limit, charged as one ``diagnostic``
        call to this binding's ledger and closed again.

        Parameters
        ----------
        field : numpy.ndarray
            Finite float64 field in V/m, shape ``(1, n_samples)``, at ``m *
            spacing``.
        spacing : float
            Positive protocol spacing in s.

        Returns
        -------
        tuple of numpy.ndarray
            ``j_tot`` and ``j_1`` in C m/s per atom on the protocol samples.

        Raises
        ------
        BackendError
            If the binding is not haCC, the field is invalid, or the call fails
            (the attempt stays charged).
        MemoryBudgetError
            If the temporary plan does not fit beside the propagation plans.
        """
        self._ensure_open()
        model = self._hacc
        if model is None:
            raise BackendError("the consistency diagnostic needs a haCC binding")
        context = _DiagnosticContext(
            model,
            self._budget,
            self._failures,
            self._provider,
            {**self._memory, "optical.host": self._host_memory_bytes},
            self._execution.memory_limit,
            archive=self._archive,
            labels={
                "configuration_identity": self._descriptor.identity,
                "binding": self._binding_record(),
            },
        )
        return _diagnostic(context, field, float(spacing))

    def record_consistency(self, record: Mapping[str, object]) -> None:
        """Attach the run's consistency provenance to the binding's metadata.

        Parameters
        ----------
        record : mapping
            JSON-compatible consistency record.
        """
        self._consistency = dict(record)

    # -- metadata ------------------------------------------------------------------

    def metadata(self) -> Mapping[str, object]:
        """Return independent JSON-compatible build, setup and call diagnostics.

        Returns
        -------
        mapping of str to object
            Scientific identity, actual plans, resource counts and failure records,
            with the model, atomic window, haCC provenance, domain flags and
            consistency record.
        """
        record: dict[str, object] = {
            "configuration_identity": self._descriptor.identity,
            "configuration": json.loads(self._descriptor.configuration_json),
            "execution": asdict(self._execution),
            "build": self._build,
            "plans": self._plan_records,
            "chunks": [asdict(chunk) for chunk in self._chunks],
            "ledger": asdict(self.ledger),
            "memory_estimate_bytes": self._estimate,
            "memory_bytes": self._memory,
            "optical_host_memory_bytes": self._host_memory_bytes,
            "diagnostic_cache_hits": self._diagnostic_cache_hits,
            "last_call": self._last_call,
            "failures": self._failures,
            "closed": self._closed,
        }
        if self._window is not None:
            record["atomic_window"] = self._window.to_json()
        if self._hacc is not None:
            record["model"] = "hacc"
            record["hacc"] = self._hacc.record()
            record["domain"] = {
                "reasons": self._domain_reasons,
                "flagged_calls": self._domain_flags,
            }
            record["consistency"] = self._consistency
        if self._archive is not None:
            record["archive"] = {
                **self._archive.reference(),
                "native_outputs": self.native_outputs,
                "trials": self._trials_opened,
            }
        return cast("dict[str, object]", _failure_safe(record))

    def memory_ledger(self) -> Mapping[str, int]:
        """Return native storage counted once and the admitted host-result reserve.

        Returns
        -------
        mapping of str to int
            Named atomic allocations in bytes, excluding separately owned optics.
        """
        return dict(self._memory)

    def close(self) -> None:
        """Close plans before setup, preserving metadata and attempted-call counts."""
        if not self._closed:
            self._budget.release_reservation()
            self._stack.close()
            self._cached = None
            self._cached_fields = None
            self._closed = True


@dataclass(frozen=True)
class _DiagnosticContext:
    """Collect what one charged haCC diagnostic call needs.

    Attributes
    ----------
    model : _HaccModel
        Open prepared model.
    budget : AtomicCallBudget
        Ledger the call is charged to, as ``diagnostic``.
    failures : list
        Failure records the call appends to.
    provider : _Provider
        Loaded provider, for error classification.
    memory : mapping
        Memory already held, in bytes.
    limit : int or None
        Memory limit in bytes.
    archive : TDSEArchiveWriter or None
        Archive session receiving the call, or None.
    labels : mapping
        The run's identities for the archive record: configuration identity (None
        for a standalone check) and the build and execution records.
    """

    model: _HaccModel
    budget: AtomicCallBudget
    failures: list[object]
    provider: _Native
    memory: Mapping[str, int]
    limit: int | None
    archive: TDSEArchiveWriter | None = None
    labels: Mapping[str, object] = dataclasses.field(default_factory=dict)


def _diagnostic(
    context: _DiagnosticContext,
    field: Float[np.ndarray, "1 n_samples"],
    spacing: float,
) -> tuple[Float[np.ndarray, " n_samples"], Float[np.ndarray, " n_samples"]]:
    """Run one charged one-row haCC diagnostic on a temporary plan.

    Parameters
    ----------
    context : _DiagnosticContext
        Model, ledger, failure log, provider and memory admission.
    field : numpy.ndarray
        Finite float64 field in V/m, shape ``(1, n_samples)``.
    spacing : float
        Protocol spacing in s.

    Returns
    -------
    tuple of numpy.ndarray
        Owning ``j_tot`` and ``j_1`` in C m/s.
    """
    model, budget, provider = context.model, context.budget, context.provider
    if (
        not isinstance(field, np.ndarray)
        or field.dtype != np.dtype(np.float64)
        or field.ndim != 2
        or field.shape[0] != 1
        or field.shape[1] < 4
        or not np.all(np.isfinite(field))
    ):
        raise BackendError("the consistency field must be finite float64 (1, samples)")
    samples = int(field.shape[1])
    with ExitStack() as stack:
        try:
            plan = stack.enter_context(
                model.plan(rows=1, spacing=spacing, samples=samples)
            )
            model.check_info(plan.info, rows=1, samples=samples, spacing=spacing)
            extra = {
                f"diagnostic.{entry.name}": entry.bytes
                for entry in plan.memory_ledger().entries
                if not entry.shared
            }
        except SolpropError:
            raise
        except Exception as error:
            kind = _classify(error, provider)
            raise BackendError(
                f"TDSE diagnostic preparation failed: {error} [{kind}]"
            ) from error
        # The temporary plan overlaps the propagation plans: admit their sum.
        _admit_memory(
            {**context.memory, **extra, "diagnostic.host": 8 * 20 * samples},
            context.limit,
        )
        budget.reserve(1, calls=1)
        try:
            drive = np.ascontiguousarray(field)
            evaluation = _diagnostic_archive_begin(context, drive, spacing)
            budget.begin_evaluation(purpose="diagnostic", calls=1)
            budget.charge(purpose="diagnostic")
            raw: _Native = None
            archived: TDSEArchiveError | None = None
            try:
                raw = plan.evaluate(drive)
                archived = _diagnostic_archive_result(context, evaluation, raw)
                binding_view = _DiagnosticView(model, provider, spacing)
                binding_view.validate(raw, samples)
                total, first = binding_view.currents(raw, samples)
            except Exception as error:
                budget.record_failure()
                kind = _classify(error, provider)
                context.failures.append(
                    _failure_safe(
                        {
                            "purpose": "diagnostic",
                            "classification": kind,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                )
                wrapped = BackendError(
                    f"TDSE consistency diagnostic failed: {error} [{kind}]"
                )
                if archived is not None:
                    wrapped.add_note(
                        f"the TDSE archive could not store this result: {archived}"
                    )
                _diagnostic_archive_failure(context, evaluation, error, kind, wrapped)
                raise wrapped from error
            if archived is not None:
                raise archived
            if context.archive is not None and evaluation is not None:
                context.archive.finish_evaluation(evaluation, "returned")
        finally:
            budget.release_reservation()
    return (
        np.array(total[0], dtype=np.float64, copy=True),
        np.array(first[0], dtype=np.float64, copy=True),
    )


def _diagnostic_archive_begin(
    context: _DiagnosticContext,
    field: Float[np.ndarray, "1 n_samples"],
    spacing: float,
) -> int | None:
    """Write a consistency diagnostic's labels and input before its provider call.

    Parameters
    ----------
    context : _DiagnosticContext
        The call's context, with its archive.
    field : numpy.ndarray
        The exact protocol field passed, V/m.
    spacing : float
        Protocol spacing in s.

    Returns
    -------
    int or None
        The archive's evaluation identifier, or None without an archive.
    """
    archive = context.archive
    if archive is None:
        return None
    archive.ensure_writable()
    model = context.model
    samples = int(field.shape[1])
    labels = dict(context.labels)
    binding = labels.pop("binding", {})
    return archive.begin_evaluation(
        {
            "trial_id": None,
            "role": "diagnostic",
            "purpose": "consistency",
            "z": None,
            "density": None,
            "input_identity": waveform_identity(field),
            "basis_identity": model.basis.identity,
            "model": "hacc",
            "descriptor_outputs": "total_current",
            "native_outputs": model.outputs,
            "archive_outputs": archive.outputs,
            # The consistency protocol's own lattice: provider time zero is its
            # first sample, which is not a time of the optical coordinate.
            "lattice": "consistency_protocol",
            "lattice_samples": samples,
            "spacing": float(spacing),
            "optical_time_origin": None,
            "window_first": 0,
            "window_last": samples - 1,
            "provider_samples": samples,
            "provider_time_zero": 0.0,
            **labels,
        },
        prewindow=field,
        drives=[(0, 1, field)],
        gate=None,
        binding=cast("Mapping[str, object]", binding),
    )


def _diagnostic_archive_result(
    context: _DiagnosticContext, evaluation: int | None, raw: _Native
) -> TDSEArchiveError | None:
    """Store a consistency diagnostic's returned record before its validation.

    Parameters
    ----------
    context : _DiagnosticContext
        The call's context.
    evaluation : int or None
        The archive's evaluation, or None.
    raw : object
        The provider's record, as returned.

    Returns
    -------
    TDSEArchiveError or None
        The write failure, if any.
    """
    archive = context.archive
    if archive is None or evaluation is None:
        return None
    outputs = context.model.outputs
    assert outputs is not None
    try:
        archive.write_result(evaluation, 0, raw, record_serializer("hacc", outputs))
    except TDSEArchiveError as error:
        return error
    return None


def _diagnostic_archive_failure(
    context: _DiagnosticContext,
    evaluation: int | None,
    error: BaseException,
    classification: str,
    primary: BaseException,
) -> None:
    """Record a failed consistency diagnostic, keeping the provider error primary.

    Parameters
    ----------
    context : _DiagnosticContext
        The call's context.
    evaluation : int or None
        The archive's evaluation, or None.
    error : BaseException
        The exception.
    classification : str
        Its failure class.
    primary : BaseException
        The error raised, which gains a note if recording fails.
    """
    archive = context.archive
    if archive is None or evaluation is None:
        return
    try:
        archive.write_failure(
            evaluation, chunk=0, error=error, classification=classification
        )
        archive.finish_evaluation(evaluation, "raised")
    except TDSEArchiveError as failure:
        primary.add_note(f"the TDSE archive could not record this failure: {failure}")


class _DiagnosticView:
    """Validate a one-row diagnostic result with the binding's haCC rules.

    Parameters
    ----------
    model : _HaccModel
        Prepared model.
    provider : _Provider
        Loaded provider.
    spacing : float
        Protocol spacing in s.
    """

    def __init__(self, model: _HaccModel, provider: _Provider, spacing: float) -> None:
        self._model = model
        self._provider = provider
        self._spacing = spacing

    def validate(self, raw: _Native, samples: int) -> None:
        """Check the bound, monitors, outcomes and validity of one row.

        Parameters
        ----------
        raw : object
            Provider result.
        samples : int
            Protocol samples.
        """
        bound = raw.neglected_current_bound
        if bound is None:
            if self._model.closure:
                raise BackendError("TDSE neglected-current bounds are malformed")
        elif (
            not isinstance(bound, np.ndarray)
            or bound.dtype != np.dtype(np.float64)
            or bound.shape != (1, samples)
            or not np.all(np.isfinite(bound))
            or np.any(bound < 0)
        ):
            raise BackendError("TDSE neglected-current bounds are malformed")
        if len(raw.outcomes) != 1 or not raw.outcomes[0].ok or not np.all(raw.valid):
            raise BackendError("TDSE provider reported a failed diagnostic row")

    def currents(self, raw: _Native, samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the validated total and first-order currents.

        Parameters
        ----------
        raw : object
            Provider result.
        samples : int
            Protocol samples.

        Returns
        -------
        tuple of numpy.ndarray
            ``(1, samples)`` currents in C m/s.
        """
        model = self._model
        model.check_info(raw.info, rows=1, samples=samples, spacing=self._spacing)
        if model.outputs == "total_current":
            if (
                not isinstance(raw, model.hacc.TotalCurrentResponse)
                or raw.shadow_orders != (1,)
                or raw.shadow_total_current is None
                or raw.shadow_total_current.shape != (1, 1, samples)
            ):
                raise BackendError("haCC compact diagnostic result is malformed")
            return raw.total_current, raw.shadow_total_current[:, 0, :]
        if (
            not isinstance(raw, model.hacc.GridResponse)
            or raw.shadow is None
            or raw.shadow.orders != (1,)
        ):
            raise BackendError("haCC full diagnostic result is malformed")
        first = raw.shadow.component("total_current" if model.closure else "current", 1)
        return raw.total_current, first


class DiagnosticSession:
    """Prepare a haCC model alone for a standalone consistency check.

    Used by :func:`solprop.driver.tdse_consistency.check_consistency`, which needs
    one diagnostic row and no propagation plans. Close it when done.

    Parameters
    ----------
    basis : HaccBasis
        Validated preset or custom record.
    execution : TDSEExecution
        Device, memory limit and the explicit call budget of the check.
    archive : TDSEArchiveWriter or None, optional
        Archive session receiving the diagnostic call, labelled ``diagnostic``.
    """

    def __init__(
        self,
        basis: HaccBasis,
        execution: TDSEExecution,
        *,
        archive: TDSEArchiveWriter | None = None,
    ) -> None:
        self._execution = execution
        self._archive = archive
        self._budget = AtomicCallBudget(execution.max_atomic_calls, 1)
        self._failures: list[object] = []
        self._stack = ExitStack()
        self._provider = _load_provider()
        self.build = _build_record(self._provider, execution)
        try:
            self._model = _HaccModel(
                self._provider,
                basis,
                execution,
                self._stack,
                archive="required" if archive is None else archive.outputs,
            )
            self._memory = dict(self._model.memory)
            if archive is not None:
                self._memory.update(archive.memory_ledger(rows=1, samples=1))
            _admit_memory(self._memory, execution.memory_limit)
        except BaseException as error:
            self._stack.close()
            if not isinstance(error, Exception) or isinstance(error, SolpropError):
                raise
            kind = _classify(error, self._provider)
            raise BackendError(f"TDSE preparation failed: {error} [{kind}]") from error

    @property
    def ledger(self) -> AtomicCallLedger:
        """Return the check's attempted-call accounting.

        Returns
        -------
        AtomicCallLedger
            Diagnostic charges, including failures.
        """
        return self._budget.snapshot()

    def diagnostic_currents(
        self, field: Float[np.ndarray, "1 n_samples"], spacing: float
    ) -> tuple[Float[np.ndarray, " n_samples"], Float[np.ndarray, " n_samples"]]:
        """Evaluate one charged diagnostic row on the prepared model.

        Parameters
        ----------
        field : numpy.ndarray
            Finite float64 field in V/m, shape ``(1, n_samples)``.
        spacing : float
            Protocol spacing in s.

        Returns
        -------
        tuple of numpy.ndarray
            ``j_tot`` and ``j_1`` in C m/s per atom.
        """
        context = _DiagnosticContext(
            self._model,
            self._budget,
            self._failures,
            self._provider,
            self._memory,
            self._execution.memory_limit,
            archive=self._archive,
            labels={
                "configuration_identity": None,
                "binding": {
                    "build": self.build,
                    "execution": asdict(self._execution),
                    "hacc": self._model.record(),
                },
            },
        )
        return _diagnostic(context, field, float(spacing))

    def metadata(self) -> Mapping[str, object]:
        """Return the session's model, build, ledger and failure records.

        Returns
        -------
        mapping
            JSON-compatible provenance.
        """
        return cast(
            "dict[str, object]",
            _failure_safe(
                {
                    "hacc": self._model.record(),
                    "build": self.build,
                    "ledger": asdict(self.ledger),
                    "failures": self._failures,
                    "memory_bytes": self._memory,
                    **(
                        {}
                        if self._archive is None
                        else {"archive": self._archive.reference()}
                    ),
                }
            ),
        )

    def close(self) -> None:
        """Close the model; repeated calls are harmless."""
        self._stack.close()

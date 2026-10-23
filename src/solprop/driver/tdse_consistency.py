# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check a haCC model's weak-field linear response against optmat's helium (SEP 20).

Because the haCC current is the whole atom's and nothing rescales it, every haCC
configuration is checked before propagation (SEP 20 sections 4.8, 4.9 and 5.4). A
fixed synthetic weak pulse drives one row; the per-atom first-order dipole
``p_1 = j_1 / (i omega)`` gives the model's susceptibility
``alpha_haCC = p_1 / (eps0 E)`` in m^3, and the ratio
``kappa_1 = alpha_optmat / alpha_haCC`` is formed over the supported band. Its largest
``|kappa_1 - 1|`` passes, warns or refuses; an invalid extraction always refuses. The
measured values are cached per configuration and provider revision, never per build.

The cubic ratio kappa_3 is evidence only (validation level L3), extracted by
:func:`cubic_evidence` over the fundamental band and never part of the gate.

Fourier convention. Inside this module a hat is NumPy's forward DFT, under which
``d/dt`` becomes ``+i omega`` (the internal convention of
``docs/dev/conventions.md`` section 2). optmat's susceptibility is in the optics
convention, so its complex value is conjugated before comparison; the real and
imaginary parts are compared together, never the real part alone.
"""

import hashlib
import json
import logging
import math
import os
import tempfile
import time
import uuid
import warnings
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Complex, Float

import solprop
import solprop.driver.tdse as tdse_runtime
from solprop.constants import EPS0, C
from solprop.exceptions import (
    BackendError,
    TDSEConsistencyError,
    TDSEConsistencyWarning,
)
from solprop.nonlinear.tdse import (
    HaccBasis,
    TDSEConsistency,
    TDSEExecution,
    TDSEGasResponse,
)
from solprop.output.tdse_archive import TDSEArchive, TDSEArchiveWriter
from solprop.physdata import species
from solprop.translate.tdse import AtomicCallLedger

__all__ = [
    "PROTOCOL_VERSION",
    "ConsistencyResult",
    "CubicEvidence",
    "DiagnosticSource",
    "GateBinding",
    "LinearReduction",
    "apply_outcome",
    "cache_directory",
    "check_consistency",
    "cubic_evidence",
    "describe",
    "evaluate_consistency",
    "gate_run",
    "protocol_field",
    "reduce_linear",
]

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
"""Version of the frozen kappa_1 protocol below; any change invalidates every cache."""

# The frozen protocol, version 1 (SEP 20 section 4.9, Q16). A 4 fs intensity-FWHM
# Gaussian at the run's reference wavelength, cosine carrier phase at the centre of a
# 40 fs window, 1e11 W/cm^2 peak, sampled every 25 as. At 5 FWHM from the centre the
# field is exp(-2 ln2 25) ~ 9e-16 of its peak, so window truncation is far below the
# endpoint-leakage limit. The 25 as spacing keeps the provider's cubic interpolation
# error, (omega h)^4 / 24 * 9/16, below 4e-6 at the band's upper edge (4.6e15 rad/s).
_FWHM = 4e-15
_WINDOW = 40e-15
_SPACING = 25e-18
_INTENSITY = 1e15
# The comparison band: |E_hat| above 1e-3 of its maximum, positive frequency, inside
# optmat's declared support. Below 1e-3 the division amplifies the response's
# rounding and the extraction no longer measures the model.
_BAND_FLOOR = 1e-3
# Endpoint leakage: the field and j_1 over the first and last 2% of the window must
# stay below 1e-6 of their peaks, or the finite record is not a transient.
_EDGE_FRACTION = 0.02
_LEAKAGE_LIMIT = 1e-6
# A spectral bin of j_1 must exceed 64 eps times the L1 norm of j_1: the transform's
# rounding bound (log2 of the size times eps times the L1 norm) with margin.
_ROUNDING_FACTOR = 64.0
# Cubic evidence (not gated); these limits are policy, chosen as follows. The
# fundamental band is |E_hat| above
# 1e-1 of its maximum, where the third-harmonic spectrum of the 4 fs pulse is below
# 1e-6 of its own peak. At 1e11 W/cm^2 successive odd orders of the remainder differ
# by about 1e-4, so the c3 + c5 model misses the lambda = 4 row only by its seventh
# order, 11520 c7 / r4 ~ 1e-6; a 1e-4 limit is therefore exceeded only by a resolved
# departure from perturbative scaling (or a first-order leak above about 4e-5 of c3).
_FUNDAMENTAL_FLOOR = 1e-1
_LADDER_LIMIT = 1e-4

Outcome = Literal["passed", "warned", "refused", "invalid", "skipped"]


def protocol_field(
    reference_wavelength: float,
) -> tuple[float, Float[np.ndarray, " n_samples"]]:
    """Return the protocol's spacing and synthetic weak field (version 1).

    Parameters
    ----------
    reference_wavelength : float
        Carrier wavelength in m, positive and finite.

    Returns
    -------
    spacing : float
        Sample spacing in s; sample ``m`` lies at ``m * spacing``.
    field : numpy.ndarray
        Real field in V/m, shape ``(1601,)``, over a 40 fs window.

    Raises
    ------
    BackendError
        If the wavelength is not positive and finite.
    """
    if not (math.isfinite(reference_wavelength) and reference_wavelength > 0):
        raise BackendError("the consistency check needs a positive finite wavelength")
    samples = round(_WINDOW / _SPACING) + 1
    times = np.arange(samples, dtype=np.float64) * _SPACING
    centre = 0.5 * (samples - 1) * _SPACING
    amplitude = math.sqrt(2 * _INTENSITY / (EPS0 * C))
    omega0 = 2 * math.pi * C / reference_wavelength
    shifted = times - centre
    envelope = np.exp(-2 * math.log(2) * (shifted / _FWHM) ** 2)
    return _SPACING, amplitude * envelope * np.cos(omega0 * shifted)


@dataclass(frozen=True)
class LinearReduction:
    """Hold one kappa_1 extraction and its validity (SEP 20 section 4.8).

    Attributes
    ----------
    valid : bool
        Whether every validity condition held.
    reason : str or None
        Why the extraction is invalid, or None.
    wavelength : tuple of float
        Band wavelengths in m, ascending frequency.
    kappa_real, kappa_imag : tuple of float
        ``kappa_1 = alpha_optmat / alpha_haCC`` on the band.
    deviation : float or None
        Largest ``|kappa_1 - 1|`` on the band.
    worst_wavelength : float or None
        Wavelength in m of that largest deviation.
    worst_kappa : tuple of float or None
        ``(Re, Im)`` of kappa_1 there.
    """

    valid: bool
    reason: str | None
    wavelength: tuple[float, ...] = ()
    kappa_real: tuple[float, ...] = ()
    kappa_imag: tuple[float, ...] = ()
    deviation: float | None = None
    worst_wavelength: float | None = None
    worst_kappa: tuple[float, float] | None = None


def _invalid(reason: str) -> LinearReduction:
    """Return an invalid reduction with its reason.

    Parameters
    ----------
    reason : str
        The failed validity condition.

    Returns
    -------
    LinearReduction
        Invalid record with no values.
    """
    return LinearReduction(False, reason)


def _edge_ratio(values: Float[np.ndarray, " n_samples"]) -> float:
    """Return the largest endpoint magnitude relative to the peak.

    Parameters
    ----------
    values : numpy.ndarray
        A finite record with a nonzero peak.

    Returns
    -------
    float
        ``max |x|`` over the first and last edge fraction, over ``max |x|``.
    """
    edge = max(1, round(_EDGE_FRACTION * values.size))
    ends = np.concatenate([values[:edge], values[-edge:]])
    return float(np.max(np.abs(ends)) / np.max(np.abs(values)))


def _record_reason(
    drive: Float[np.ndarray, " n_samples"],
    first_order: Float[np.ndarray, " n_current"],
) -> str | None:
    """Return why the time records cannot be reduced, or None.

    Parameters
    ----------
    drive : numpy.ndarray
        Field in V/m.
    first_order : numpy.ndarray
        First-order current in C m/s.

    Returns
    -------
    str or None
        The failed condition: shape or finiteness, a zero record, endpoint leakage.
    """
    if (
        drive.shape != first_order.shape
        or drive.ndim != 1
        or drive.size < 4
        or not np.all(np.isfinite(drive))
        or not np.all(np.isfinite(first_order))
    ):
        return "the field and current must be finite records of one shape"
    if not np.any(first_order) or not np.any(drive):
        return "the first-order current or the field is identically zero"
    for name, values in (("field", drive), ("first-order current", first_order)):
        ratio = _edge_ratio(values)
        if ratio > _LEAKAGE_LIMIT:
            return (
                f"endpoint leakage of the {name}: {ratio:.3g} of its peak exceeds "
                f"{_LEAKAGE_LIMIT:g}"
            )
    return None


def reduce_linear(
    spacing: float,
    drive: Float[np.ndarray, " n_samples"],
    first_order: Float[np.ndarray, " n_current"],
    reference: Callable[[Float[np.ndarray, " n_band"]], Complex[np.ndarray, " n_band"]],
    support: tuple[float, float],
) -> LinearReduction:
    """Form kappa_1 from a first-order current and a reference susceptibility.

    Parameters
    ----------
    spacing : float
        Sample spacing in s.
    drive : numpy.ndarray
        Real field in V/m.
    first_order : numpy.ndarray
        The model's first-order current per atom, C m/s, same shape.
    reference : callable
        Per-atom susceptibility in m^3 at angular frequencies in rad/s, in the
        optics convention (conjugated here).
    support : tuple of float
        Lowest and highest angular frequency in rad/s where the reference is valid.

    Returns
    -------
    LinearReduction
        The kappa_1 values over the band, or an invalid record naming the failed
        condition:
        non-finite or mismatched input, a zero current, endpoint leakage, an empty
        band, a band bin below the rounding floor, or a non-finite ratio. DC is
        always excluded from the division by ``i omega``.
    """
    reason = _record_reason(drive, first_order)
    if reason is not None:
        return _invalid(reason)
    field_hat = np.fft.rfft(drive)
    current_hat = np.fft.rfft(first_order)
    omega = 2 * math.pi * np.fft.rfftfreq(drive.size, spacing)
    band = (
        (omega > 0)
        & (np.abs(field_hat) > _BAND_FLOOR * np.max(np.abs(field_hat)))
        & (omega >= support[0])
        & (omega <= support[1])
    )
    if not np.any(band):
        return _invalid("the supported comparison band is empty")
    floor = _ROUNDING_FACTOR * np.finfo(float).eps * float(np.sum(np.abs(first_order)))
    if np.any(np.abs(current_hat[band]) <= floor):
        return _invalid("the first-order current is below its rounding floor in band")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        model = current_hat[band] / (1j * omega[band] * EPS0 * field_hat[band])
        kappa = np.conj(reference(omega[band])) / model
    if not np.all(np.isfinite(kappa)):
        return _invalid("kappa_1 is non-finite on the band")
    deviation = np.abs(kappa - 1)
    worst = int(np.argmax(deviation))
    wavelength = 2 * math.pi * C / omega[band]
    return LinearReduction(
        True,
        None,
        tuple(float(value) for value in wavelength),
        tuple(float(value) for value in kappa.real),
        tuple(float(value) for value in kappa.imag),
        float(deviation[worst]),
        float(wavelength[worst]),
        (float(kappa[worst].real), float(kappa[worst].imag)),
    )


def _helium_reference() -> tuple[
    Callable[[Float[np.ndarray, " n_band"]], Complex[np.ndarray, " n_band"]],
    tuple[float, float],
    dict[str, object],
]:
    """Return optmat helium's per-atom susceptibility, its support and identity.

    Returns
    -------
    tuple
        The susceptibility function of angular frequency, the supported angular
        frequency interval in rad/s, and the material identity record.
    """
    helium = species("He")
    susceptibility = helium.susceptibility
    shortest, longest = susceptibility.wavelength_range
    support = (
        0.0 if math.isinf(longest) else 2 * math.pi * C / longest,
        2 * math.pi * C / shortest,
    )
    identity = {
        "material": "He",
        "records": [
            {"name": record.name, "digest": record.digest}
            for record in helium.material_records
            if record.kind == "gas_refractive"
        ],
        "optmat_version": helium.material_records[0].optmat_version,
    }

    def reference(
        omega: Float[np.ndarray, " n_band"],
    ) -> Complex[np.ndarray, " n_band"]:
        """Evaluate optmat's per-atom susceptibility in m^3.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.

        Returns
        -------
        numpy.ndarray
            Complex susceptibility in the optics convention.
        """
        return np.asarray(susceptibility.susceptibility(omega), dtype=np.complex128)

    return reference, support, identity


def cache_directory() -> Path:
    """Return the consistency cache directory (SEP 20 section 6.5).

    Returns
    -------
    pathlib.Path
        ``$SOLPROP_CACHE_DIR/tdse-consistency`` when that is set, else
        ``$XDG_CACHE_HOME/solprop/tdse-consistency``, else
        ``~/.cache/solprop/tdse-consistency``. Deleting it is always safe.
    """
    explicit = os.environ.get("SOLPROP_CACHE_DIR")
    if explicit:
        return Path(explicit) / "tdse-consistency"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "solprop" / "tdse-consistency"


def _key(inputs: Mapping[str, object]) -> str:
    """Digest the cache key's canonical inputs.

    Parameters
    ----------
    inputs : mapping
        Basis identity, provider revision, material identity, wavelength, protocol.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _reduction_from(record: object) -> LinearReduction | None:
    """Rebuild a cached reduction, or None if malformed, non-finite or invalid.

    Parameters
    ----------
    record : object
        Parsed ``reduction`` record of a cache entry.

    Returns
    -------
    LinearReduction or None
        A valid reduction with matching finite arrays, else None.
    """
    # Explicit checks, not assertions: a cached entry is untrusted input, and
    # ``python -O`` would strip an assertion and admit a malformed entry.
    if not isinstance(record, dict):
        return None
    try:
        reduction = LinearReduction(
            valid=record["valid"],
            reason=record["reason"],
            wavelength=tuple(record["wavelength"]),
            kappa_real=tuple(record["kappa_real"]),
            kappa_imag=tuple(record["kappa_imag"]),
            deviation=record["deviation"],
            worst_wavelength=record["worst_wavelength"],
            worst_kappa=tuple(record["worst_kappa"]),
        )
    except KeyError, TypeError, ValueError:
        return None
    return reduction if _cached_reduction_valid(reduction) else None


def _finite_floats(values: tuple[object, ...]) -> bool:
    """Return whether every value is a finite float.

    Parameters
    ----------
    values : tuple
        Candidate values.

    Returns
    -------
    bool
        True when each is a ``float`` and finite.
    """
    return all(isinstance(value, float) and math.isfinite(value) for value in values)


def _cached_reduction_valid(reduction: LinearReduction) -> bool:
    """Check that a cached reduction is a complete, finite, self-consistent valid one.

    Parameters
    ----------
    reduction : LinearReduction
        A reduction rebuilt from a cache entry.

    Returns
    -------
    bool
        True only when it is marked valid without a reason, its three arrays are
        nonempty, equally long and finite, and its summary (deviation, worst
        wavelength and worst kappa) is finite and agrees with the arrays.
    """
    size = len(reduction.wavelength)
    worst_kappa = reduction.worst_kappa
    if (
        reduction.valid is not True
        or reduction.reason is not None
        or size == 0
        or len(reduction.kappa_real) != size
        or len(reduction.kappa_imag) != size
        or worst_kappa is None
        or len(worst_kappa) != 2
        or not _finite_floats(
            (
                *reduction.wavelength,
                *reduction.kappa_real,
                *reduction.kappa_imag,
                reduction.deviation,
                reduction.worst_wavelength,
                *worst_kappa,
            )
        )
    ):
        return False
    kappa = np.array(reduction.kappa_real) + 1j * np.array(reduction.kappa_imag)
    deviation = np.abs(kappa - 1)
    worst = int(np.argmax(deviation))
    return (
        reduction.deviation == float(deviation[worst])
        and reduction.worst_wavelength == reduction.wavelength[worst]
        and worst_kappa == (reduction.kappa_real[worst], reduction.kappa_imag[worst])
    )


def _read_cache(
    path: Path, inputs: Mapping[str, object]
) -> tuple[LinearReduction, dict[str, object]] | None:
    """Return a cache entry whose inputs, values and validity check out.

    Parameters
    ----------
    path : pathlib.Path
        Entry file.
    inputs : mapping
        The key's inputs, which the entry must repeat exactly.

    Returns
    -------
    tuple or None
        The validated reduction and its computation record, or None when absent or
        unusable (logged, never a pass).
    """
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        logger.warning(
            "ignoring unreadable consistency cache entry %s: %s", path, error
        )
        return None
    if not isinstance(entry, dict) or entry.get("inputs") != dict(inputs):
        logger.warning("ignoring consistency cache entry %s: inputs disagree", path)
        return None
    reduction = _reduction_from(entry.get("reduction"))
    computed = entry.get("computed")
    if reduction is None or not isinstance(computed, dict):
        logger.warning("ignoring consistency cache entry %s: invalid values", path)
        return None
    return reduction, cast("dict[str, object]", computed)


def _write_cache(path: Path, entry: Mapping[str, object]) -> None:
    """Write a cache entry atomically: a temporary file, then a rename.

    Parameters
    ----------
    path : pathlib.Path
        Final entry path.
    entry : mapping
        JSON-compatible entry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(entry, stream, sort_keys=True, allow_nan=False)
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ConsistencyResult:
    """Record one haCC consistency decision and its provenance (SEP 20, 5.4).

    Attributes
    ----------
    outcome : {'passed', 'warned', 'refused', 'invalid', 'skipped'}
        The decision.
    model : str
        The model's label, for example ``haCC(1)`` or ``haCC(5) (custom)``.
    preset : str or None
        The provider preset identity, or None for a custom basis.
    reduction : LinearReduction or None
        The kappa_1 extraction; None when skipped.
    protocol_version : int
        Version of the frozen protocol.
    cache_key : str or None
        The cache entry's key; None when skipped.
    from_cache : bool
        Whether the reduction was read from the cache.
    computed : Mapping or None
        Execution space, build and threads of the computation that produced the
        values, with its wall and CPU time.
    warn, refuse : float
        The bounds applied.
    enforce : bool
        Whether the check ran.
    ledger : AtomicCallLedger or None
        Calls charged by this check (zero on a cache hit); None when the check ran
        inside a run, whose ledger holds them.
    cubic_evidence : Mapping or None
        For a preset, the reference to its L3 kappa_3 evidence; never gated.
    """

    outcome: Outcome
    model: str
    preset: str | None
    reduction: LinearReduction | None
    protocol_version: int
    cache_key: str | None
    from_cache: bool
    computed: Mapping[str, object] | None
    warn: float
    refuse: float
    enforce: bool
    ledger: AtomicCallLedger | None = None
    cubic_evidence: Mapping[str, object] | None = field(default=None)

    def to_record(self) -> dict[str, object]:
        """Return the JSON-compatible record stored under ``tdse.consistency``.

        Returns
        -------
        dict
            Plain values.
        """
        record = asdict(self)
        record["computed"] = None if self.computed is None else dict(self.computed)
        record["cubic_evidence"] = (
            None if self.cubic_evidence is None else dict(self.cubic_evidence)
        )
        # Plain JSON containers (lists, not tuples): the form a file reads back.
        return json.loads(json.dumps(record, allow_nan=False))


@runtime_checkable
class DiagnosticSource(Protocol):
    """Run one charged diagnostic row on an open haCC model."""

    def diagnostic_currents(
        self, field: Float[np.ndarray, "1 n_samples"], spacing: float
    ) -> tuple[Float[np.ndarray, " n_samples"], Float[np.ndarray, " n_samples"]]:
        """Return ``j_tot`` and ``j_1`` for one row.

        Parameters
        ----------
        field : numpy.ndarray
            Field in V/m, shape ``(1, n_samples)``.
        spacing : float
            Spacing in s.

        Returns
        -------
        tuple of numpy.ndarray
            Currents in C m/s.
        """
        ...


@runtime_checkable
class GateBinding(DiagnosticSource, Protocol):
    """A run's binding as the gate needs it: a diagnostic row and its record."""

    def metadata(self) -> Mapping[str, object]:
        """Return the binding's provenance, including its native build.

        Returns
        -------
        Mapping[str, object]
            JSON-compatible record.
        """
        ...

    def record_consistency(self, record: Mapping[str, object]) -> None:
        """Attach the consistency record to the binding's provenance.

        Parameters
        ----------
        record : Mapping[str, object]
            JSON-compatible consistency record.
        """
        ...

    @property
    def plans_built(self) -> bool:
        """Return whether the propagation plans exist.

        Returns
        -------
        bool
            False while a deferred binding awaits :meth:`build_plans`.
        """
        ...

    def build_plans(self) -> None:
        """Build the deferred propagation plans after the check."""
        ...


def _decide(deviation: float, policy: TDSEConsistency) -> Outcome:
    """Apply the bounds to a valid ``|kappa_1 - 1|`` (rules 2 to 4 of 5.4).

    Parameters
    ----------
    deviation : float
        Largest ``|kappa_1 - 1|``.
    policy : TDSEConsistency
        The bounds.

    Returns
    -------
    {'passed', 'warned', 'refused'}
        Passed at or below ``warn``, warned up to and including ``refuse``.
    """
    if deviation <= policy.warn:
        return "passed"
    if deviation <= policy.refuse:
        return "warned"
    return "refused"


def evaluate_consistency(  # noqa: PLR0913 - the check's complete context
    source: DiagnosticSource | None,
    basis: HaccBasis,
    policy: TDSEConsistency,
    *,
    reference_wavelength: float,
    provider_revision: str,
    computed_on: Mapping[str, object],
    refresh: bool = False,
) -> ConsistencyResult:
    """Obtain kappa_1 from the cache or by one diagnostic call, then decide.

    Parameters
    ----------
    source : object or None
        Open haCC binding or session; None only when the check is skipped.
    basis : HaccBasis
        The configuration checked.
    policy : TDSEConsistency
        The decision bounds and switch.
    reference_wavelength : float
        Carrier of the protocol pulse in m.
    provider_revision : str
        Provider source revision; part of the cache key.
    computed_on : mapping
        Execution space, build fingerprint and threads; recorded, not keyed.
    refresh : bool, optional
        Recompute and overwrite the cache entry.

    Returns
    -------
    ConsistencyResult
        The decision with its provenance. Nothing is raised or warned here.
    """
    common = {
        "model": basis.label,
        "preset": basis.preset,
        "protocol_version": PROTOCOL_VERSION,
        "warn": policy.warn,
        "refuse": policy.refuse,
        "enforce": policy.enforce,
        "cubic_evidence": None
        if basis.preset is None
        else {
            "preset": basis.preset,
            "kappa_3": "SEP 20 validation level L3 (reported, not gated)",
        },
    }
    if not policy.enforce:
        return ConsistencyResult(
            "skipped",
            reduction=None,
            cache_key=None,
            from_cache=False,
            computed=None,
            **common,  # pyright: ignore[reportArgumentType] - typed above
        )
    reference, support, material = _helium_reference()
    inputs = {
        "basis_identity": basis.identity,
        "provider_revision": provider_revision,
        "material": material,
        "reference_wavelength": float(reference_wavelength),
        "protocol_version": PROTOCOL_VERSION,
    }
    key = _key(inputs)
    path = cache_directory() / f"{key}.json"
    cached = None if refresh else _read_cache(path, inputs)
    if cached is not None:
        reduction, computed = cached
        return ConsistencyResult(
            _decide(_deviation(reduction.deviation), policy),
            reduction=reduction,
            cache_key=key,
            from_cache=True,
            computed=computed,
            **common,  # pyright: ignore[reportArgumentType] - typed above
        )
    if source is None:
        raise BackendError("an enforced consistency check needs an atomic model")
    spacing, drive = protocol_field(reference_wavelength)
    wall, cpu = time.perf_counter(), time.process_time()
    _, first = source.diagnostic_currents(drive[None, :], spacing)
    computed = {
        **computed_on,
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
        "created_unix": time.time(),
        "solprop_version": solprop.__version__,
    }
    reduction = reduce_linear(spacing, drive, first, reference, support)
    if not reduction.valid:
        return ConsistencyResult(
            "invalid",
            reduction=reduction,
            cache_key=key,
            from_cache=False,
            computed=computed,
            **common,  # pyright: ignore[reportArgumentType] - typed above
        )
    _write_cache(
        path,
        {"inputs": inputs, "reduction": asdict(reduction), "computed": computed},
    )
    return ConsistencyResult(
        _decide(_deviation(reduction.deviation), policy),
        reduction=reduction,
        cache_key=key,
        from_cache=False,
        computed=computed,
        **common,  # pyright: ignore[reportArgumentType] - typed above
    )


def _deviation(value: float | None) -> float:
    """Return a value known to be a float.

    Parameters
    ----------
    value : float or None
        A valid reduction's deviation.

    Returns
    -------
    float
        The same value.

    Raises
    ------
    BackendError
        If the value is missing, which a valid reduction never allows.
    """
    if value is None:
        raise BackendError("a valid kappa_1 reduction has no deviation")
    return value


def describe(result: ConsistencyResult) -> str:
    """Return the human-readable summary used in warnings and refusals.

    Parameters
    ----------
    result : ConsistencyResult
        A decided, warned, refused or invalid result.

    Returns
    -------
    str
        Model, worst wavelength and kappa_1, and the bounds.
    """
    reduction = result.reduction
    if (
        reduction is None
        or not reduction.valid
        or reduction.worst_kappa is None
        or reduction.worst_wavelength is None
    ):
        reason = "skipped" if reduction is None else reduction.reason
        return f"{result.model}: the kappa_1 extraction is invalid ({reason})"
    real, imag = reduction.worst_kappa
    return (
        f"{result.model}: |kappa_1 - 1| = {reduction.deviation:.4g} at "
        f"{reduction.worst_wavelength * 1e9:.1f} nm (kappa_1 = {real:.6g}"
        f"{imag:+.3g}j) against warn={result.warn:g}, refuse={result.refuse:g}"
    )


def apply_outcome(result: ConsistencyResult) -> None:
    """Warn for a warned or skipped result, raise for a refused or invalid one.

    Parameters
    ----------
    result : ConsistencyResult
        The decided result.

    Raises
    ------
    TDSEConsistencyError
        For ``refused`` and ``invalid``, with the result attached as ``result``.
    """
    if result.outcome == "skipped":
        warnings.warn(
            f"{result.model}: the weak-field consistency check was skipped "
            "(TDSEConsistency(enforce=False)); the run's linear response is unchecked",
            TDSEConsistencyWarning,
            stacklevel=3,
        )
    elif result.outcome == "warned":
        warnings.warn(describe(result), TDSEConsistencyWarning, stacklevel=3)
    elif result.outcome in ("refused", "invalid"):
        hint = (
            "; widen TDSEConsistency(refuse=...) to admit it, or skip the check with "
            "TDSEConsistency(enforce=False)"
            if result.outcome == "refused"
            else "; an invalid extraction is refused whatever the bounds, and "
            "TDSEConsistency(enforce=False) skips the check"
        )
        error = TDSEConsistencyError(
            f"haCC consistency check {result.outcome}: {describe(result)}{hint}"
        )
        error.result = result
        raise error


def gate_run(
    binding: DiagnosticSource,
    response: TDSEGasResponse,
    *,
    reference_wavelength: float,
    provider_revision: str,
    computed_on: Mapping[str, object],
) -> ConsistencyResult:
    """Run the gate before a haCC run's first propagation call.

    Parameters
    ----------
    binding : object
        The run's open binding; a computed check is charged to its ledger.
    response : TDSEGasResponse
        The declared haCC response, carrying the policy.
    reference_wavelength : float
        The run's reference wavelength in m.
    provider_revision : str
        Pinned provider revision.
    computed_on : mapping
        The binding's execution space, build and threads.

    Returns
    -------
    ConsistencyResult
        The decision; ``ledger`` is None because the run's ledger holds the call.
        The caller records it and then calls :func:`apply_outcome`.
    """
    basis = response.basis
    if not isinstance(basis, HaccBasis):
        raise BackendError("the consistency gate applies to haCC responses only")
    return evaluate_consistency(
        binding,
        basis,
        response.consistency,
        reference_wavelength=reference_wavelength,
        provider_revision=provider_revision,
        computed_on=computed_on,
    )


def check_consistency(
    response: TDSEGasResponse,
    execution: TDSEExecution,
    *,
    reference_wavelength: float,
    refresh: bool = False,
    tdse_archive: TDSEArchive | None = None,
) -> ConsistencyResult:
    """Run the haCC weak-field linear check on its own, without a propagation.

    The same protocol and cache as a run's gate (SEP 20 sections 4.9 and 6.5): a
    cache hit makes no atomic call; a miss prepares the model alone, makes one
    one-row ``diagnostic`` call within ``execution``'s budget and memory limit, and
    writes the entry. No warning or refusal is raised here: the outcome is returned.
    With `tdse_archive`, the call is archived with the ``diagnostic`` role in a new
    archive of its own (SEP 20 section 5.8); a cache hit writes no evaluation.

    Parameters
    ----------
    response : TDSEGasResponse
        A haCC response; its ``consistency`` policy supplies the bounds.
    execution : TDSEExecution
        Device, memory limit and explicit call budget of the check.
    reference_wavelength : float
        Carrier wavelength of the protocol pulse in m (the run's lambda0).
    refresh : bool, optional
        Recompute and overwrite an existing cache entry.
    tdse_archive : solprop.output.TDSEArchive or None, optional
        Archive the provider call in this new HDF5 file; an existing file is
        refused, never overwritten.

    Returns
    -------
    ConsistencyResult
        Outcome, kappa_1 values, cache use, provenance and the check's own ledger.

    Raises
    ------
    BackendError
        If the response is not haCC, or the provider is missing, stale or fails.
    TDSEArchiveError
        If the archive's path exists, or a write to it fails.
    """
    basis = response.basis
    if not isinstance(basis, HaccBasis):
        raise BackendError("check_consistency applies to haCC responses only")
    if tdse_archive is not None and not isinstance(tdse_archive, TDSEArchive):
        raise BackendError("tdse_archive must be a TDSEArchive or None")
    session = None
    archive = None
    try:
        if response.consistency.enforce:
            archive = _open_check_archive(tdse_archive, basis, execution)
            session = tdse_runtime.DiagnosticSession(basis, execution, archive=archive)
        build = {} if session is None else session.build
        native = build.get("native", {})
        result = evaluate_consistency(
            session,
            basis,
            response.consistency,
            reference_wavelength=reference_wavelength,
            provider_revision=tdse_runtime.pinned_revision(),
            computed_on={
                "execution_space": native.get("execution_space")
                if isinstance(native, dict)
                else None,
                "build": build,
                "threads": native.get("requested_threads")
                if isinstance(native, dict)
                else None,
            },
            refresh=refresh,
        )
        ledger = None if session is None else session.ledger
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            if archive is not None:
                archive.close()
    return ConsistencyResult(
        **{**_fields_of(result), "ledger": ledger},
    )


def _open_check_archive(
    archive: TDSEArchive | None, basis: HaccBasis, execution: TDSEExecution
) -> TDSEArchiveWriter | None:
    """Open a new archive for a standalone check, before any atomic work.

    Parameters
    ----------
    archive : TDSEArchive or None
        The request.
    basis : HaccBasis
        The checked basis.
    execution : TDSEExecution
        The check's resources.

    Returns
    -------
    TDSEArchiveWriter or None
        The session, or None without a request.
    """
    if archive is None:
        return None
    return TDSEArchiveWriter.open(
        archive,
        run_uuid=str(uuid.uuid4()),
        kind="consistency_check",
        optical=None,
        context={
            "basis": basis.identity,
            "label": basis.label,
            "protocol": PROTOCOL_VERSION,
            "execution": asdict(execution),
            "provider_pin": tdse_runtime.pinned_revision(),
        },
    )


def _fields_of(result: ConsistencyResult) -> dict[str, object]:
    """Return a result's fields without converting nested records.

    Parameters
    ----------
    result : ConsistencyResult
        A result.

    Returns
    -------
    dict
        Field name to value.
    """
    return {name: getattr(result, name) for name in result.__dataclass_fields__}


@dataclass(frozen=True)
class CubicEvidence:
    """Hold the L3 kappa_3 evidence of one model (SEP 20 section 4.8; not gated).

    Attributes
    ----------
    kappa_3 : float
        Real least-squares scale with ``F[j3_ref] ~ kappa_3 F[c3_hat]`` over the
        fundamental band, ``j3_ref = eps0 gamma3 d/dt(E^3)``.
    residual : float
        ``||F[j3_ref] - kappa_3 F[c3_hat]|| / ||F[j3_ref]||`` on that band.
    ladder_residual : float
        ``||r_4 - (64 c3_hat + 1024 c5_hat)|| / ||r_4||``: how well the c3 + c5
        model predicts the lambda = 4 row.
    ladder_ok : bool
        Whether ``ladder_residual`` is at most the stated bound, 1e-4.
    leakage : float
        ``||c_1|| / ||c_3||`` of the three-term fit ``r_a = a c1 + a^3 c3 + a^5 c5``:
        the shadow's first-order leakage relative to the cubic term (Q15).
    seventh_order_floor : float
        The fit's floor for that leakage, relative to ``||c_3||``: a seventh order
        ``c7`` enters the fitted ``c1`` as exactly ``64 c7``, which three rows cannot
        separate from a leak. It is estimated from the measured order ratio as
        ``64 (||c5_hat|| / ||c3_hat||)^2``; a ``leakage`` at or below it is not
        resolved.
    leakage_resolved : bool
        Whether ``leakage`` exceeds ``seventh_order_floor``.
    band_wavelength : tuple of float
        The fundamental band's wavelength limits in m.
    """

    kappa_3: float
    residual: float
    ladder_residual: float
    ladder_ok: bool
    leakage: float
    seventh_order_floor: float
    leakage_resolved: bool
    band_wavelength: tuple[float, float]


def cubic_evidence(
    spacing: float,
    drive: Float[np.ndarray, " n_samples"],
    remainders: Mapping[int, Float[np.ndarray, " n_samples"]],
    gamma3: float,
) -> CubicEvidence:
    """Extract kappa_3 and the leakage figure from an amplitude ladder (L3).

    With the remainders ``r_a = j_tot - j_1`` of the same base field scaled by
    ``a in {1, 2, 4}``, ``c3_hat = (32 r_1 - r_2) / 24 = c3 - 4 c7 + ...`` and
    ``c5_hat = (r_2 - 8 r_1) / 24`` neglect the first-order leakage ``c1``. kappa_3 is
    compared over the fundamental band only, where the third-harmonic component of
    the cubic response is negligible.

    Parameters
    ----------
    spacing : float
        Sample spacing in s.
    drive : numpy.ndarray
        The base (``a = 1``) field in V/m.
    remainders : mapping
        ``{1: r_1, 2: r_2, 4: r_4}`` in C m/s per atom, each like ``drive``.
    gamma3 : float
        The per-atom Kerr coefficient of optmat at the carrier, m^5/V^2.

    Returns
    -------
    CubicEvidence
        The reported, never gated, evidence.

    Raises
    ------
    BackendError
        If the ladder is incomplete, mis-shaped, non-finite or degenerate.
    """
    if set(remainders) != {1, 2, 4} or any(
        values.shape != drive.shape or not np.all(np.isfinite(values))
        for values in remainders.values()
    ):
        raise BackendError("the cubic ladder needs finite r_1, r_2 and r_4")
    r1, r2, r4 = remainders[1], remainders[2], remainders[4]
    c3_hat = (32 * r1 - r2) / 24
    c5_hat = (r2 - 8 * r1) / 24
    ladder = float(np.linalg.norm(r4 - (64 * c3_hat + 1024 * c5_hat)))
    norm4 = float(np.linalg.norm(r4))
    matrix = np.array([[1.0, 1.0, 1.0], [2.0, 8.0, 32.0], [4.0, 64.0, 1024.0]])
    c1, c3, _ = np.linalg.solve(matrix, np.stack([r1, r2, r4]))
    field_hat = np.fft.rfft(drive)
    omega = 2 * math.pi * np.fft.rfftfreq(drive.size, spacing)
    band = (omega > 0) & (
        np.abs(field_hat) >= _FUNDAMENTAL_FLOOR * np.max(np.abs(field_hat))
    )
    reference = (1j * omega * EPS0 * gamma3 * np.fft.rfft(drive**3))[band]
    model = np.fft.rfft(c3_hat)[band]
    reference_norm = float(np.linalg.norm(reference))
    cubic_norm = float(np.linalg.norm(c3))
    model_norm = float(np.linalg.norm(model))
    if not (norm4 > 0 and reference_norm > 0 and cubic_norm > 0 and model_norm > 0):
        raise BackendError("the cubic ladder is degenerate")
    scale = float(np.sum(np.conj(model) * reference).real / model_norm**2)
    wavelengths = 2 * math.pi * C / omega[band]
    leakage = float(np.linalg.norm(c1)) / cubic_norm
    # The fitted c1 absorbs 64 c7 exactly (the solve of the 1, 2, 4 ladder against
    # a^7 gives (64, -84, 21) on (c1, c3, c5)); c7 is extrapolated geometrically from
    # the measured ratio of the fifth to the third order.
    order_ratio = float(np.linalg.norm(c5_hat)) / float(np.linalg.norm(c3_hat))
    floor = 64 * order_ratio**2
    return CubicEvidence(
        kappa_3=scale,
        residual=float(np.linalg.norm(reference - scale * model)) / reference_norm,
        ladder_residual=ladder / norm4,
        ladder_ok=ladder / norm4 <= _LADDER_LIMIT,
        leakage=leakage,
        seventh_order_floor=floor,
        leakage_resolved=leakage > floor,
        band_wavelength=(float(np.min(wavelengths)), float(np.max(wavelengths))),
    )

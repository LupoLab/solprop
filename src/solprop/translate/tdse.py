# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Define the provider-free host contract for the helium TDSE response.

The seam holds immutable descriptors and owning SI arrays. The driver owns the
optional native runtime, stable row chunks and transactional call reservations.
No physical subtraction, calibration or density factor is applied here.
"""

import functools
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import KW_ONLY, asdict, dataclass, field, replace
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Bool, Float

from solprop.exceptions import (
    BackendError,
    ResponseBudgetError,
    SolpropError,
    TranslationError,
)
from solprop.grid import (
    Collar,
    FieldGrid,
    TimeGrid,
    planck_taper,
    planck_taper_derivative,
    planck_taper_derivative_error,
)
from solprop.nonlinear.tdse import (
    HaccBasis,
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEConsistency,
    TDSEGasResponse,
    TDSEKerrModel,
    TDSEModel,
    TDSESampling,
)

__all__ = [
    "DESCRIPTOR_SCHEMA",
    "HACC_NORM_GROWTH_LIMIT",
    "AtomicCallBudget",
    "AtomicCallLedger",
    "AtomicWindowRecord",
    "TDSEBinding",
    "TDSEDescriptor",
    "TDSEPurpose",
    "TDSEResult",
    "lattice_times",
    "refine_response_grid",
    "resolve_atomic_window",
    "waveform_identity",
]

HACC_NORM_GROWTH_LIMIT = 2.0
"""The provider's gross-failure norm-growth limit pinned in every haCC setup record."""

TDSEPurpose = Literal["startup", "predictor", "corrected", "diagnostic"]
"""The relation of an evaluated field to the optical integration or diagnostics."""
_PURPOSES = ("startup", "predictor", "corrected", "diagnostic")


def refine_response_grid(grid: TimeGrid, sampling: TDSESampling) -> FieldGrid:
    """Refine the fine lattice while preserving every coarse coordinate and collar.

    Parameters
    ----------
    grid : TimeGrid
        Existing field grid, with its already realized physical time window.
    sampling : TDSESampling
        Power-of-two refinement of the existing fine sample count.

    Returns
    -------
    FieldGrid
        Validated grid with unchanged coarse axes, period and collars. A factor
        of one returns the original grid. No duplicate endpoint is appended.

    Raises
    ------
    TranslationError
        If an envelope or an inconsistent grid is supplied.
    """
    if not isinstance(grid, FieldGrid):
        raise TranslationError("TDSE response sampling requires a FieldGrid")
    grid.validate()
    if sampling.refinement == 1:
        return grid
    samples = grid.n_to * sampling.refinement
    spacing = grid.dt_fine / sampling.refinement
    if not math.isfinite(spacing) or spacing <= 0:
        raise TranslationError("TDSE response refinement underflows its spacing")
    refined = replace(
        grid,
        t_fine=(np.arange(samples, dtype=float) - samples / 2) * spacing,
        omega_fine=np.arange(samples // 2 + 1, dtype=float) * grid.d_omega,
        dt_fine=spacing,
        ffac=grid.ffac * sampling.refinement,
    )
    refined.validate()
    return refined


def lattice_times(samples: int, spacing: float) -> Float[np.ndarray, " n_to"]:
    """Return the fine response lattice's sample times exactly as the grid forms them.

    Parameters
    ----------
    samples : int
        Positive even number of fine samples.
    spacing : float
        Positive fine spacing in s.

    Returns
    -------
    numpy.ndarray
        ``(n - samples / 2) * spacing`` in s, the same arithmetic as
        :meth:`solprop.grid.FieldGrid.from_limits` and :func:`refine_response_grid`.
    """
    return (np.arange(samples, dtype=float) - samples / 2) * spacing


@dataclass(frozen=True)
class AtomicWindowRecord:
    """Hold an atomic window resolved on one fine response lattice (SEP 20, 5.3).

    Attributes
    ----------
    duration : float
        Nominal duration ``T_a`` in s.
    taper : float
        Width of each inner Planck taper in s.
    start, stop : float
        Nominal bounds ``t_- = -T_a / 2`` and ``t_+ = T_a / 2`` in s.
    first, last : int
        Inclusive lattice indices of ``t_s``, the first sample at or after ``t_-``,
        and ``t_e``, the last at or before ``t_+``.
    t_s, t_e : float
        Times of those samples in s; provider time zero is ``t_s``.
    gate : tuple of float
        Planck breakpoints ``(t_-, t_- + w, t_+ - w, t_+)`` in s, never stretched to
        the sample endpoints.
    """

    duration: float
    taper: float
    start: float
    stop: float
    first: int
    last: int
    t_s: float
    t_e: float
    gate: tuple[float, float, float, float]

    @property
    def samples(self) -> int:
        """Return the number of retained samples ``last - first + 1``.

        Returns
        -------
        int
            Atomic samples passed to the provider.
        """
        return self.last - self.first + 1

    @property
    def atomic_duration(self) -> float:
        """Return the realised atomic duration ``t_e - t_s``.

        Returns
        -------
        float
            Duration in s, compared with a preset's tested window.
        """
        return self.t_e - self.t_s

    def gate_values(
        self, times: Float[np.ndarray, " n_to"]
    ) -> Float[np.ndarray, " n_to"]:
        """Evaluate the window's gate ``g(t)`` at absolute lattice times.

        Parameters
        ----------
        times : numpy.ndarray
            Times in s, shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            ``g`` in ``[0, 1]``: zero outside ``[t_-, t_+]``, one between the tapers.
        """
        return planck_taper(times, *self.gate)

    def gate_derivative(
        self, times: Float[np.ndarray, " n_to"]
    ) -> Float[np.ndarray, " n_to"]:
        """Evaluate the analytic derivative ``g'(t)`` of the window's gate.

        Parameters
        ----------
        times : numpy.ndarray
            Times in s, shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            ``dg/dt`` in 1/s: positive on the leading taper, negative on the
            trailing one, zero elsewhere (:func:`solprop.grid.planck_taper_derivative`).
        """
        return planck_taper_derivative(times, *self.gate)

    def to_json(self) -> dict[str, object]:
        """Return the primitive record stored in a descriptor's lattice.

        Returns
        -------
        dict
            Plain JSON values: SI times and integer indices.
        """
        return {**asdict(self), "gate": list(self.gate)}


def resolve_atomic_window(
    window: TDSEAtomicWindow, samples: int, spacing: float
) -> AtomicWindowRecord:
    """Resolve a declared window on a fine response lattice.

    Parameters
    ----------
    window : TDSEAtomicWindow
        Declared duration and taper in s; the duration must be resolved.
    samples : int
        Fine samples of the full optical lattice.
    spacing : float
        Fine spacing in s.

    Returns
    -------
    AtomicWindowRecord
        Sample range, endpoint times and gate breakpoints.

    Raises
    ------
    TranslationError
        If the duration is unresolved or exceeds the lattice period, or the window
        holds fewer than the provider's four samples.
    """
    if window.duration is None:
        raise TranslationError(
            "the atomic window's duration is unresolved: give "
            "TDSEAtomicWindow(duration=...) or build through a capillary entry point "
            "with trange"
        )
    period = samples * spacing
    if window.duration > period:
        raise TranslationError(
            f"atomic window duration {window.duration:g} s exceeds the optical "
            f"lattice period {period:g} s"
        )
    times = lattice_times(samples, spacing)
    start, stop = -window.duration / 2, window.duration / 2
    inside = np.flatnonzero((times >= start) & (times <= stop))
    if inside.size < 4:
        raise TranslationError(
            "the atomic window holds fewer than the provider's four samples"
        )
    first, last = int(inside[0]), int(inside[-1])
    return AtomicWindowRecord(
        duration=window.duration,
        taper=window.taper,
        start=start,
        stop=stop,
        first=first,
        last=last,
        t_s=float(times[first]),
        t_e=float(times[last]),
        gate=(start, start + window.taper, stop - window.taper, stop),
    )


def _basis_from_json(values: Mapping[str, object]) -> TDSEBasis | HaccBasis:
    """Rebuild either basis type from its primitive record.

    Parameters
    ----------
    values : mapping
        A basis record; a haCC record carries ``kind="hacc_helium"``.

    Returns
    -------
    TDSEBasis or HaccBasis
        The revalidated record.
    """
    basis = dict(values)
    if basis.pop("kind", None) == "hacc_helium":
        for name in ("radial_edges", "radial_orders"):
            basis[name] = tuple(cast("list[object]", basis[name]))
        return HaccBasis(**cast("dict[str, Any]", basis))
    for name in ("radial_edges", "radial_orders", "cuts"):
        basis[name] = tuple(cast("list[object]", basis[name]))
    return TDSEBasis(**cast("dict[str, Any]", basis))


def _response_from_json(encoded: str) -> TDSEGasResponse:
    """Revalidate persisted primitive science at the single dynamic JSON boundary.

    Parameters
    ----------
    encoded : str
        Descriptor JSON containing a primitive response record in SI units.

    Returns
    -------
    TDSEGasResponse
        Reconstructed frozen specification with all scientific validation applied.
        The consistency policy is never persisted, so it takes its default.
    """
    record = json.loads(encoded)["response"]
    calibration = dict(record["calibration"])
    for name in ("wavelength_range", "limitations"):
        calibration[name] = tuple(calibration[name])
    window = record.get("atomic_window")
    return TDSEGasResponse(
        **{
            **record,
            "basis": _basis_from_json(record["basis"]),
            "calibration": TDSECalibration(**calibration),
            "sampling": TDSESampling(**record["sampling"]),
            "atomic_window": None if window is None else TDSEAtomicWindow(**window),
            "consistency": TDSEConsistency(),
        }
    )


DESCRIPTOR_SCHEMA = "solprop.tdse-descriptor.v5"
"""The one TDSE descriptor schema solprop writes and reads.

Its response record carries every field of the response but the consistency policy:
``model`` and ``atomic_window`` (``null`` for the full lattice), with a ``kind``-tagged
haCC basis. Version 5 is SEP 26's version-4 layout without the ``source_filter`` key:
the source form follows from ``kerr_model``, the native closure's source being always
the edge-free gated one. Earlier versions are not read.
"""


def _response_record(response: TDSEGasResponse) -> dict[str, object]:
    """Return a response's primitive descriptor record.

    Parameters
    ----------
    response : TDSEGasResponse
        Validated scientific specification.

    Returns
    -------
    dict
        The response record; the consistency policy never enters it.
    """
    record = asdict(response)
    del record["consistency"]
    if isinstance(response.basis, HaccBasis):
        record["basis"] = {"kind": "hacc_helium", **record["basis"]}
    return record


def _validate_schema(encoded: str) -> None:
    """Require the one descriptor schema, its complete response record and lattice.

    Parameters
    ----------
    encoded : str
        Descriptor JSON with SI temporal-collar breakpoints.
    """
    configuration = json.loads(encoded)
    response = configuration["response"]
    if configuration.get("schema") != DESCRIPTOR_SCHEMA:
        raise TranslationError(
            f"unsupported TDSE descriptor schema; only {DESCRIPTOR_SCHEMA} is read"
        )
    if not {"model", "atomic_window"} <= response.keys():
        raise TranslationError("TDSE descriptor records model and atomic_window")
    _validated_collar(configuration)
    if configuration["lattice"]["tail_policy"] != "unchanged":
        raise TranslationError("TDSE raw-current tail_policy must remain unchanged")


def _validated_collar(configuration: Mapping[str, Any]) -> Collar:
    """Return a descriptor's recorded temporal collar after checking its order.

    Parameters
    ----------
    configuration : mapping
        Parsed descriptor JSON.

    Returns
    -------
    Collar
        The fine-lattice collar in s.
    """
    collar = Collar(**configuration["lattice"]["time_collar"])
    if not collar.zero_low < collar.one_low <= collar.one_high < collar.zero_high:
        raise TranslationError("TDSE source time_collar breakpoints are invalid")
    return collar


def _setup(samples: int, spacing: float) -> dict[str, object]:
    """Return the admitted SAE provider ground-state setup for one atomic lattice.

    Parameters
    ----------
    samples : int
        Atomic samples: the full lattice, or the atomic window's.
    spacing : float
        Response spacing in s.

    Returns
    -------
    dict
        Setup policy. The growing-mode window is the atomic run duration in s,
        so the provider checks mode growth over exactly the propagated time.
    """
    return {
        "initial_l": 0,
        "initial_index": 0,
        "growing": "remove",
        "growth_tolerance": 1e-3,
        "window_duration": (samples - 1) * spacing,
        "accept_removed_component": False,
    }


def _hacc_setup() -> dict[str, object]:
    """Return the pinned haCC provider setup record (SEP 20, 6.2).

    Returns
    -------
    dict
        Every row restarts from the field-free ground root at each evaluation; a
        growing root below the cut is refused at preparation; the gross-failure
        norm-growth limit is pinned.
    """
    return {
        "initial": "ground_root",
        "restart": "every_evaluation",
        "growing": "refuse_below_cut",
        "norm_growth_limit": HACC_NORM_GROWTH_LIMIT,
    }


@functools.cache
def _window_from_json(encoded: str) -> AtomicWindowRecord | None:
    """Parse a descriptor's resolved atomic window once per configuration.

    Parameters
    ----------
    encoded : str
        Validated descriptor JSON.

    Returns
    -------
    AtomicWindowRecord or None
        The resolved window, or None for the full lattice.
    """
    record = json.loads(encoded)["lattice"].get("atomic_window")
    if record is None:
        return None
    return AtomicWindowRecord(**{**record, "gate": tuple(record["gate"])})


@dataclass(frozen=True)
class TDSEDescriptor:
    """Carry a resolved response identity and primitive values across the seam.

    Construct with :meth:`from_response` after resolving the fine grid. Native
    handles and physics objects are absent; ``configuration_json`` is sufficient
    to reconstruct the declared public provider basis during preparation.

    Attributes
    ----------
    configuration_json : str
        Canonical JSON scientific specification, resolved lattice and pinned
        provider setup policy. Quantities are SI unless dimensionless.
    identity : str
        SHA-256 of that exact JSON string.
    kerr_model : str
        Cubic-source policy.
    remainder_scale : float
        Dimensionless remainder scale, applied by source assembly only.
    cubic_scale : float or None
        Dimensionless third-order dipole scale, or absent.
    shadow_order : int
        Requested highest shadow order, 1 or 3.
    outputs : {'full', 'total_current'}
        Minimal required provider result contract.
    samples : int
        Response samples, equal to the resolved fine FFT size.
    spacing : float
        Positive response spacing in s.
    time_origin : float
        Optical label in s of provider time zero; no sample values are shifted.
    max_atomic_step : float
        Requested positive maximum atomic step in s.
    initial_l, initial_index : int
        Prepared ground-state selection, both zero.
    growing : str
        Pinned provider growing-mode policy, ``remove``.
    growth_tolerance : float
        Pinned provider default dimensionless threshold, 1e-3.
    window_duration : float or None
        Provider setup's declared growing-mode window in s, the atomic run
        duration ``(samples - 1) * spacing``. None only for a bare descriptor
        without a configuration; admission requires the run duration.
    accept_removed_component : bool
        False: preparation must satisfy the provider's removed-component check.

    Every descriptor has the one schema :data:`DESCRIPTOR_SCHEMA`: the complete
    response record (``model``, a ``kind``-tagged haCC basis and ``atomic_window``), the
    exact SI collar breakpoints in ``lattice.time_collar``, ``lattice.atomic_window``
    when a window is set (its gate replacing the collar), and, for haCC, the pinned
    ``setup`` record of SEP 20 section 6.2. Raw atomic histories retain
    ``tail_policy="unchanged"``. A native descriptor's source is gated and edge-free
    (:attr:`gated`); its gate is admitted by the steady-current certificate when the
    problem is validated.
    """

    configuration_json: str = field(repr=False)
    identity: str
    kerr_model: TDSEKerrModel
    remainder_scale: float
    cubic_scale: float | None
    shadow_order: int
    outputs: Literal["full", "total_current"]
    samples: int
    spacing: float
    time_origin: float
    max_atomic_step: float
    initial_l: int = 0
    initial_index: int = 0
    growing: str = "remove"
    growth_tolerance: float = 1e-3
    window_duration: float | None = None
    accept_removed_component: bool = False

    def __post_init__(self) -> None:
        """Protect the seam identity against independently modified scalar fields."""
        digest = hashlib.sha256(self.configuration_json.encode()).hexdigest()
        if digest != self.identity:
            raise TranslationError(
                "TDSE descriptor identity does not match its configuration"
            )
        try:
            _metadata_json(self.configuration_json, "TDSE descriptor", dict)
            configuration = json.loads(self.configuration_json)
            _validate_schema(self.configuration_json)
            response_spec = _response_from_json(self.configuration_json)
            response = configuration["response"]
            lattice = configuration["lattice"]
            setup = configuration["setup"]
            order = 1 if response_spec.kerr_model == "tdse_native" else 3
            outputs = (
                "full" if response_spec.kerr_model == "tdse_shadow" else "total_current"
            )
            if self.shadow_order != order or self.outputs != outputs:
                raise TranslationError(
                    "TDSE model requires a different order/output policy"
                )
            window = _resolved_window(
                response_spec, lattice["samples"], lattice["spacing"]
            )
            if (None if window is None else window.to_json()) != lattice.get(
                "atomic_window"
            ):
                raise TranslationError(
                    "TDSE descriptor atomic window disagrees with its declaration"
                )
            expected_setup = (
                _hacc_setup()
                if response_spec.model == "hacc"
                else _setup(
                    lattice["samples"] if window is None else window.samples,
                    lattice["spacing"],
                )
            )
            if setup != expected_setup:
                raise TranslationError(
                    "TDSE descriptor changed its admitted ground-state setup"
                )
            if type(self.samples) is not int or self.samples < 4:
                raise TranslationError("TDSE descriptor requires at least four samples")
            if not all(
                math.isfinite(value) and value > 0
                for value in (
                    self.spacing,
                    self.max_atomic_step,
                )
            ) or not math.isfinite(self.time_origin):
                raise TranslationError("TDSE descriptor has an invalid time lattice")
            expected = {
                "kerr_model": response["kerr_model"],
                "remainder_scale": response["remainder_scale"],
                "cubic_scale": response["cubic_scale"],
                "max_atomic_step": response["max_atomic_step"],
                "shadow_order": configuration["shadow_order"],
                "outputs": configuration["outputs"],
                **{
                    name: lattice[name]
                    for name in ("samples", "spacing", "time_origin")
                },
                **asdict(_setup_fields(response_spec.model, setup)),
            }
        except (KeyError, TypeError, ValueError, SolpropError) as error:
            raise TranslationError(
                f"TDSE descriptor configuration is invalid: {error}"
            ) from error
        if any(
            getattr(self, name, object()) != value for name, value in expected.items()
        ):
            raise TranslationError(
                "TDSE descriptor values disagree with its configuration"
            )

    @property
    def gated(self) -> bool:
        """Return whether the source is gated and edge-free (SEP 26).

        Returns
        -------
        bool
            True for the native closure, whose remainder current is gated by the
            window or grid collar with the edge term removed; False for the cubic
            replacements, whose sources are never gated.
        """
        return self.kerr_model == "tdse_native"

    @property
    def model(self) -> TDSEModel:
        """Return the atomic model family recorded in the configuration.

        Returns
        -------
        {'sae', 'hacc'}
            The recorded ``response.model``.
        """
        return self.response.model

    @property
    def response(self) -> TDSEGasResponse:
        """Return the revalidated scientific response this descriptor carries.

        Returns
        -------
        TDSEGasResponse
            Reconstructed record; its consistency policy is the default, because
            policy never enters a descriptor.
        """
        return _cached_response(self.configuration_json)

    @property
    def atomic_window(self) -> AtomicWindowRecord | None:
        """Return the resolved atomic window, or None for the full lattice.

        Returns
        -------
        AtomicWindowRecord or None
            Sample range, endpoint times and gate breakpoints in s.
        """
        return _window_from_json(self.configuration_json)

    def source_gate(self) -> Float[np.ndarray, " n_to"] | None:
        """Return the window's gate on the full lattice, or None without a window.

        Returns
        -------
        numpy.ndarray or None
            Read-only ``g(t_n)``, shape ``(samples,)``, evaluated at the absolute
            lattice times of :func:`lattice_times`.
        """
        return _gate_from_json(self.configuration_json)

    def source_breakpoints(self) -> tuple[float, float, float, float]:
        """Return the Planck breakpoints of the gate a gated source uses.

        Returns
        -------
        tuple of float
            ``(z_-, o_-, o_+, z_+)`` in s: the atomic window's gate when one is set,
            else the recorded fine temporal collar. Never moved to samples.
        """
        return _breakpoints_from_json(self.configuration_json)

    def source_gate_derivative(self) -> Float[np.ndarray, " n_to"]:
        """Return the analytic derivative of the gated source's gate on the lattice.

        Returns
        -------
        numpy.ndarray
            Read-only ``g'(t_n)`` in 1/s, shape ``(samples,)``, of the gate of
            :meth:`source_breakpoints`, at the absolute lattice times
            (:func:`solprop.grid.planck_taper_derivative`).
        """
        return _gate_derivative_from_json(self.configuration_json)

    def source_gate_derivative_error(self) -> Float[np.ndarray, " n_to"]:
        """Return the rounding bound of :meth:`source_gate_derivative` on the lattice.

        Returns
        -------
        numpy.ndarray
            Read-only nonnegative bound in 1/s, shape ``(samples,)``
            (:func:`solprop.grid.planck_taper_derivative_error`).
        """
        return _gate_derivative_from_json(self.configuration_json, error=True)

    @classmethod
    def from_response(
        cls, response: TDSEGasResponse, grid: FieldGrid
    ) -> "TDSEDescriptor":  # noqa: UP037 - beartype resolves this during class creation.
        """Resolve one scientific response on an already refined field grid.

        Parameters
        ----------
        response : TDSEGasResponse
            Validated scientific specification. An atomic window must have a
            resolved duration.
        grid : FieldGrid
            Result of :func:`refine_response_grid`, used by source transforms.

        Returns
        -------
        TDSEDescriptor
            Immutable descriptor with no imported provider or native handle.

        Raises
        ------
        TranslationError
            If the window's duration is unresolved, exceeds the lattice period or
            holds fewer than four samples, or the grid's fine times are not the
            standard lattice.
        """
        grid.validate()
        order = 1 if response.kerr_model == "tdse_native" else 3
        outputs: Literal["full", "total_current"] = (
            "full" if response.kerr_model == "tdse_shadow" else "total_current"
        )
        window = _resolved_window(response, grid.n_to, grid.dt_fine)
        if (
            window is not None or response.kerr_model == "tdse_native"
        ) and not np.array_equal(grid.t_fine, lattice_times(grid.n_to, grid.dt_fine)):
            raise TranslationError(
                "an atomic window or a gated native source requires the standard "
                "fine response lattice"
            )
        if response.model == "hacc":
            setup = _hacc_setup()
        else:
            setup = _setup(
                grid.n_to if window is None else window.samples, grid.dt_fine
            )
        response_record = _response_record(response)
        lattice: dict[str, object] = {
            "samples": grid.n_to,
            "spacing": grid.dt_fine,
            "time_origin": float(grid.t_fine[0]),
            "period": grid.n_to * grid.dt_fine,
            "atomic_duration": (grid.n_to - 1) * grid.dt_fine
            if window is None
            else window.atomic_duration,
            "coarse_samples": grid.n_t,
            "coarse_spacing": grid.dt,
            "reference_wavelength": grid.reference_wavelength,
            "omega_collar": asdict(grid.omega_collar),
            "time_collar": asdict(grid.t_collar),
            "interpolation": "provider_cubic",
            "tail_policy": "unchanged",
        }
        if window is not None:
            lattice["atomic_window"] = window.to_json()
        specification = {
            "schema": DESCRIPTOR_SCHEMA,
            "response": response_record,
            "setup": setup,
            "lattice": lattice,
            "shadow_order": order,
            "outputs": outputs,
        }
        encoded = json.dumps(
            specification, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return cls(
            configuration_json=encoded,
            identity=hashlib.sha256(encoded.encode()).hexdigest(),
            kerr_model=response.kerr_model,
            remainder_scale=response.remainder_scale,
            cubic_scale=response.cubic_scale,
            shadow_order=order,
            outputs=outputs,
            samples=grid.n_to,
            spacing=grid.dt_fine,
            time_origin=float(grid.t_fine[0]),
            max_atomic_step=response.max_atomic_step,
            **asdict(_setup_fields(response.model, setup)),
        )


def _resolved_window(
    response: TDSEGasResponse, samples: int, spacing: float
) -> AtomicWindowRecord | None:
    """Resolve a response's declared window on its lattice, if it has one.

    Parameters
    ----------
    response : TDSEGasResponse
        Validated specification.
    samples : int
        Fine lattice samples.
    spacing : float
        Fine spacing in s.

    Returns
    -------
    AtomicWindowRecord or None
        The resolved window, or None for the full lattice.
    """
    window = response.atomic_window
    if window is None or isinstance(window, str):
        return None
    return resolve_atomic_window(window, samples, spacing)


@dataclass(frozen=True)
class _SetupScalars:
    """Hold the descriptor's scalar provider-setup fields.

    Attributes
    ----------
    initial_l, initial_index : int
        Prepared SAE ground-state selection, both zero.
    growing : str
        Growing-mode policy: SAE ``remove``, haCC ``refuse_below_cut``.
    growth_tolerance : float
        SAE removal threshold; inert for haCC.
    window_duration : float or None
        SAE growing-mode window in s; None for haCC, which has none.
    accept_removed_component : bool
        False: preparation must satisfy the SAE removed-component check.
    """

    initial_l: int
    initial_index: int
    growing: str
    growth_tolerance: float
    window_duration: float | None
    accept_removed_component: bool


def _setup_fields(model: TDSEModel, setup: Mapping[str, object]) -> _SetupScalars:
    """Map a setup record onto the descriptor's scalar provider-setup fields.

    Parameters
    ----------
    model : {'sae', 'hacc'}
        Atomic model family.
    setup : mapping
        The configuration's setup record.

    Returns
    -------
    _SetupScalars
        The SAE record itself, or the haCC growing policy with the SAE-only fields
        at their inert defaults (no growing-mode window).
    """
    if model == "sae":
        return _SetupScalars(**cast("dict[str, Any]", setup))
    return _SetupScalars(0, 0, str(setup["growing"]), 1e-3, None, False)


@functools.cache
def _cached_response(encoded: str) -> TDSEGasResponse:
    """Rebuild a descriptor's response once per configuration.

    Parameters
    ----------
    encoded : str
        Validated descriptor JSON.

    Returns
    -------
    TDSEGasResponse
        The immutable reconstructed response.
    """
    return _response_from_json(encoded)


def _breakpoints_from_json(encoded: str) -> tuple[float, float, float, float]:
    """Return the breakpoints of a descriptor's source gate.

    Parameters
    ----------
    encoded : str
        Validated descriptor JSON.

    Returns
    -------
    tuple of float
        The window's gate when one is set, else the recorded fine collar, in s.
    """
    window = _window_from_json(encoded)
    if window is not None:
        return window.gate
    collar = _validated_collar(json.loads(encoded))
    return (collar.zero_low, collar.one_low, collar.one_high, collar.zero_high)


@functools.cache
def _gate_derivative_from_json(
    encoded: str, *, error: bool = False
) -> Float[np.ndarray, " n_to"]:
    """Evaluate a descriptor's source-gate derivative once per configuration.

    Parameters
    ----------
    encoded : str
        Validated descriptor JSON.
    error : bool, optional
        Return the derivative's rounding bound instead of the derivative.

    Returns
    -------
    numpy.ndarray
        Read-only ``g'``, or its bound, in 1/s on the full lattice: of the window
        gate, else of the recorded collar.
    """
    lattice = json.loads(encoded)["lattice"]
    times = lattice_times(lattice["samples"], lattice["spacing"])
    evaluate = planck_taper_derivative_error if error else planck_taper_derivative
    derivative = evaluate(times, *_breakpoints_from_json(encoded))
    derivative.setflags(write=False)
    return derivative


@functools.cache
def _gate_from_json(encoded: str) -> Float[np.ndarray, " n_to"] | None:
    """Evaluate a descriptor's window gate once per configuration.

    Parameters
    ----------
    encoded : str
        Validated descriptor JSON.

    Returns
    -------
    numpy.ndarray or None
        Read-only gate on the full lattice, or None without a window.
    """
    window = _window_from_json(encoded)
    if window is None:
        return None
    lattice = json.loads(encoded)["lattice"]
    gate = window.gate_values(lattice_times(lattice["samples"], lattice["spacing"]))
    gate.setflags(write=False)
    return gate


def waveform_identity(fields: Float[np.ndarray, "n_nodes n_to"]) -> str:
    """Hash the exact real host waveforms and their shape for diagnostic reuse.

    Parameters
    ----------
    fields : numpy.ndarray
        Finite float64 fields in V/m, shape ``(n_nodes, n_to)``.

    Returns
    -------
    str
        SHA-256 identity independent of input strides and writeability.

    Raises
    ------
    BackendError
        If the field is empty, nonfinite, complex or not float64.
    """
    if fields.ndim != 2 or min(fields.shape) < 1 or fields.dtype != np.dtype("float64"):
        raise BackendError(
            "TDSE fields must be a nonempty float64 (n_nodes, n_to) array"
        )
    if not np.all(np.isfinite(fields)):
        raise BackendError("TDSE fields must be finite")
    digest = hashlib.sha256()
    digest.update(str(fields.shape).encode())
    digest.update(fields.dtype.str.encode())
    digest.update(np.ascontiguousarray(fields).tobytes())
    return digest.hexdigest()


def _metadata_json(value: str, name: str, expected: type) -> None:
    """Require finite JSON containers for provider-neutral immutable diagnostics.

    Parameters
    ----------
    value : str
        Serialized candidate metadata.
    name : str
        Field name used in diagnostics.
    expected : type
        Required outer container type, either dict or list.
    """

    def reject_constant(constant: str) -> None:
        """Reject a nonstandard JSON constant.

        Parameters
        ----------
        constant : str
            Parsed nonfinite token.
        """
        raise BackendError(f"{name} contains nonfinite JSON constant {constant}")

    def finite_float(number: str) -> float:
        """Parse one finite JSON floating-point token.

        Parameters
        ----------
        number : str
            Decimal representation of a JSON number.

        Returns
        -------
        float
            Finite binary64 value.
        """
        result = float(number)
        if not math.isfinite(result):
            raise BackendError(f"{name} contains a nonfinite number")
        return result

    try:
        parsed: object = json.loads(
            value, parse_constant=reject_constant, parse_float=finite_float
        )
    except (TypeError, ValueError) as error:
        raise BackendError(f"{name} must contain valid JSON") from error
    if not isinstance(parsed, expected):
        raise BackendError(f"{name} has the wrong JSON container type")


@dataclass(frozen=True, eq=False)
class TDSEResult:
    """Own validated, read-only microscopic responses for all requested nodes.

    All current arrays are in C m/s per represented atom and the dipole is in
    C m per represented atom. No calibration or density factor is present.
    Invalid samples are refused before this record can be consumed; the adapter
    records native failure information before constructing a normalized result.

    Attributes
    ----------
    total_current : numpy.ndarray
        Completed current, float64, shape ``(n_nodes, n_to)``.
    first_order_current : numpy.ndarray
        Exact same-map first-order current with the same shape.
    input_identity : str
        Exact input-waveform identity.
    configuration_identity : str
        Resolved descriptor identity.
    valid : numpy.ndarray
        Boolean validity at every row and sample, shape ``(n_nodes, n_to)``.
    shadow_orders : tuple of int
        Available odd orders, including 1 and matching optional third-order data.
    third_order_current : numpy.ndarray or None
        Exact same-map third-order current, required for cubic replacements.
    third_order_dipole : numpy.ndarray or None
        Third-order charge dipole, required for calibrated-dipole replacement.
    monitors_json : str
        Immutable JSON object describing available numerical monitors.
    outcomes_json : str
        Immutable JSON list of per-row provider outcomes.
    """

    total_current: Float[np.ndarray, "n_nodes_total n_to_total"]
    first_order_current: Float[np.ndarray, "n_nodes_first n_to_first"]
    _: KW_ONLY
    input_identity: str
    configuration_identity: str
    valid: Bool[np.ndarray, "n_nodes_valid n_to_valid"]
    shadow_orders: tuple[int, ...]
    third_order_current: Float[np.ndarray, "n_nodes_third n_to_third"] | None = None
    third_order_dipole: Float[np.ndarray, "n_nodes_dipole n_to_dipole"] | None = None
    monitors_json: str = "{}"
    outcomes_json: str = "[]"

    def __post_init__(self) -> None:
        """Reject incomplete, nonfinite or invalid samples and take private copies."""
        shape = self.total_current.shape
        if self.total_current.ndim != 2 or min(shape) < 1:
            raise BackendError("TDSE currents must have nonempty (n_nodes, n_to) shape")
        if not self.input_identity or not self.configuration_identity:
            raise BackendError(
                "TDSE results require input and configuration identities"
            )
        if self.valid.dtype != np.dtype(bool) or self.valid.shape != shape:
            raise BackendError(
                "TDSE validity must be boolean and match every row/sample"
            )
        if not np.all(self.valid):
            row, sample = np.argwhere(~self.valid)[0]
            raise BackendError(f"TDSE response invalid at row {row}, sample {sample}")
        if (
            type(self.shadow_orders) is not tuple
            or any(type(order) is not int for order in self.shadow_orders)
            or self.shadow_orders not in ((1,), (1, 3), (1, 3, 5))
        ):
            raise BackendError("TDSE shadow_orders must identify available odd orders")
        if (3 in self.shadow_orders) != (self.third_order_current is not None):
            raise BackendError(
                "TDSE third-order current must match its available order label"
            )
        if self.third_order_dipole is not None and 3 not in self.shadow_orders:
            raise BackendError("TDSE third-order dipole requires order 3")
        for name in (
            "total_current",
            "first_order_current",
            "third_order_current",
            "third_order_dipole",
        ):
            values = getattr(self, name)
            if values is None:
                continue
            if values.dtype != np.dtype("float64") or values.shape != shape:
                raise BackendError(f"TDSE {name} must be float64 with shape {shape}")
            if not np.all(np.isfinite(values)):
                raise BackendError(f"TDSE {name} contains nonfinite samples")
            owned = np.array(values, dtype=np.float64, copy=True, order="C")
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)
        valid = np.array(self.valid, dtype=bool, copy=True, order="C")
        valid.setflags(write=False)
        object.__setattr__(self, "valid", valid)
        _metadata_json(self.monitors_json, "monitors_json", dict)
        _metadata_json(self.outcomes_json, "outcomes_json", list)


@dataclass(frozen=True)
class AtomicCallLedger:
    """Snapshot the binding's charged attempts and remaining trial reservation.

    Attributes
    ----------
    limit : int or None
        Declared maximum attempted provider calls, or unlimited.
    attempted : int
        Calls charged immediately before invocation, successful or failed.
    reserved : int
        Unspent calls held for the current whole trial.
    logical_evaluations : int
        All-node evaluations begun, counted once independently of row chunks.
    diagnostic_calls : int
        Attempted provider calls labelled as explicit diagnostics.
    failed_calls : int
        Attempted provider calls with exceptions or invalid response rows.
    """

    limit: int | None
    attempted: int
    reserved: int
    logical_evaluations: int
    diagnostic_calls: int
    failed_calls: int

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "AtomicCallLedger":  # noqa: UP037 - runtime type hook resolves the enclosing class
        """Validate a persisted atomic-call accounting record.

        Parameters
        ----------
        record : Mapping[str, object]
            Exact primitive snapshot fields, with counts in actual provider calls.

        Returns
        -------
        AtomicCallLedger
            Validated immutable snapshot. Reservations remain recorded here;
            ``AtomicCallBudget.restore`` releases them before new work.

        Raises
        ------
        BackendError
            If fields are missing or unknown, counts are not nonnegative plain
            integers, subcounts exceed attempts, or the old cap is exceeded.
        """
        count_names = (
            "attempted",
            "reserved",
            "logical_evaluations",
            "diagnostic_calls",
            "failed_calls",
        )
        if set(record) != {"limit", *count_names}:
            raise BackendError("TDSE ledger has missing or unknown fields")
        limit = record["limit"]
        if limit is not None and (type(limit) is not int or limit < 1):
            raise BackendError("TDSE ledger limit must be a positive integer or None")
        counts: list[int] = []
        for name in count_names:
            value = record[name]
            if type(value) is not int or value < 0:
                raise BackendError("TDSE ledger counts must be nonnegative integers")
            counts.append(value)
        attempted, reserved, logical, diagnostic, failed = counts
        if diagnostic > attempted or failed > attempted:
            raise BackendError("TDSE ledger subcounts exceed attempted calls")
        if limit is not None and attempted + reserved > limit:
            raise BackendError("saved TDSE ledger exceeds its original call limit")
        return cls(limit, attempted, reserved, logical, diagnostic, failed)


@dataclass
class AtomicCallBudget:
    """Reserve whole trials and charge every provider attempt before invocation.

    The binding calls ``reserve`` before a trial, ``begin_evaluation`` once for
    each all-node evaluation, and ``charge`` immediately before each chunk's
    provider call. On failure it calls ``record_failure``; a ``finally`` block
    releases unused reservations. Charges are never rolled back.

    Attributes
    ----------
    limit : int or None
        Positive maximum attempted calls, or no explicit limit.
    calls_per_evaluation : int
        Fixed number of provider chunks required for all radial rows.
    """

    limit: int | None
    calls_per_evaluation: int
    _attempted: int = field(default=0, init=False, repr=False)
    _reserved: int = field(default=0, init=False, repr=False)
    _logical_evaluations: int = field(default=0, init=False, repr=False)
    _diagnostic_calls: int = field(default=0, init=False, repr=False)
    _failed_calls: int = field(default=0, init=False, repr=False)
    _evaluation_remaining: int = field(default=0, init=False, repr=False)
    _purpose: TDSEPurpose | None = field(default=None, init=False, repr=False)
    _last_failed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate limits before any atomic resources are acquired."""
        if type(self.calls_per_evaluation) is not int or self.calls_per_evaluation < 1:
            raise TranslationError("calls_per_evaluation must be a positive integer")
        if self.limit is not None and (type(self.limit) is not int or self.limit < 1):
            raise TranslationError(
                "atomic call limit must be a positive integer or None"
            )

    def reserve(
        self, logical_calls: int, *, z: float = 0.0, calls: int | None = None
    ) -> None:
        """Reserve all chunks of the next trial before any source evaluation.

        Parameters
        ----------
        logical_calls : int
            Positive number of all-node source evaluations the trial may need.
        z : float
            Last accepted position in m, attached to an insufficient-budget error.
        calls : int or None
            Provider calls per evaluation, for a diagnostic with its own chunking
            (the haCC consistency check's single row); None uses
            ``calls_per_evaluation``.

        Raises
        ------
        ResponseBudgetError
            If the entire trial cannot be paid for. The driver attaches its state.
        BackendError
            If a reservation is already active or the count is invalid.
        """
        if self._reserved or self._evaluation_remaining:
            raise BackendError(
                "release the prior TDSE reservation before reserving a trial"
            )
        if type(logical_calls) is not int or logical_calls < 1:
            raise BackendError("logical_calls must be a positive integer")
        per_evaluation = self._per_evaluation(calls)
        required = logical_calls * per_evaluation
        if self.limit is not None and self._attempted + required > self.limit:
            raise ResponseBudgetError(
                f"TDSE atomic-call budget cannot reserve {required} calls at z={z:g}",
                z=z,
                field=None,
                calls=self._attempted,
            )
        self._reserved = required

    def _per_evaluation(self, calls: int | None) -> int:
        """Return the provider calls of one evaluation.

        Parameters
        ----------
        calls : int or None
            Explicit positive count, or None for ``calls_per_evaluation``.

        Returns
        -------
        int
            Positive number of provider calls.
        """
        if calls is None:
            return self.calls_per_evaluation
        if type(calls) is not int or calls < 1:
            raise BackendError("calls per evaluation must be a positive integer")
        return calls

    def begin_evaluation(
        self, *, purpose: TDSEPurpose, calls: int | None = None
    ) -> None:
        """Count one complete all-node request before its first provider chunk.

        Parameters
        ----------
        purpose : {'startup', 'predictor', 'corrected', 'diagnostic'}
            Relationship of the driving field to the optical step.
        calls : int or None
            Provider calls of this evaluation; None uses ``calls_per_evaluation``.

        Raises
        ------
        BackendError
            If the previous evaluation is unfinished or sufficient chunks were
            not reserved, or the purpose is unknown.
        """
        if purpose not in _PURPOSES:
            raise BackendError("unknown TDSE evaluation purpose")
        per_evaluation = self._per_evaluation(calls)
        if self._evaluation_remaining or self._reserved < per_evaluation:
            raise BackendError("TDSE evaluation requires a complete unused reservation")
        self._logical_evaluations += 1
        self._evaluation_remaining = per_evaluation
        self._purpose = purpose
        self._last_failed = True

    def charge(self, *, purpose: TDSEPurpose, z: float = 0.0) -> None:
        """Charge one reserved provider attempt immediately before invoking it.

        Parameters
        ----------
        purpose : {'startup', 'predictor', 'corrected', 'diagnostic'}
            Must match the current all-node evaluation's label.
        z : float
            Position in m, retained in a misuse diagnostic.

        Raises
        ------
        BackendError
            If the call is unreserved or carries the wrong evaluation label.
        """
        if (
            self._evaluation_remaining < 1
            or self._reserved < 1
            or purpose != self._purpose
        ):
            raise BackendError(f"unreserved or mislabelled TDSE atomic call at z={z:g}")
        self._attempted += 1
        self._reserved -= 1
        self._evaluation_remaining -= 1
        self._diagnostic_calls += int(purpose == "diagnostic")
        self._last_failed = False

    def record_failure(self) -> None:
        """Mark the last charged provider attempt failed without refunding it.

        Raises
        ------
        BackendError
            If there is no charged call or the last call was already marked failed.
        """
        if self._attempted == 0 or self._last_failed:
            raise BackendError("no unrecorded TDSE provider failure is available")
        self._failed_calls += 1
        self._last_failed = True

    def release_reservation(self) -> None:
        """Release unspent trial calls after success, rejection or an exception."""
        self._reserved = 0
        self._evaluation_remaining = 0
        self._purpose = None

    def restore(self, ledger: AtomicCallLedger) -> None:
        """Restore cumulative charges while releasing an interrupted reservation.

        The new execution configuration supplies the limit: it may explicitly
        increase the old cap, but it cannot be below calls already attempted.

        Parameters
        ----------
        ledger : AtomicCallLedger
            Snapshot saved at the previous run's last accepted checkpoint.

        Raises
        ------
        BackendError
            If counts are invalid, the new limit is insufficient, or this budget
            has already begun work. Restoring never resets cumulative charges.
        """
        if self._attempted or self._reserved or self._logical_evaluations:
            raise BackendError("restore TDSE accounting before beginning new work")
        values = (
            ledger.attempted,
            ledger.reserved,
            ledger.logical_evaluations,
            ledger.diagnostic_calls,
            ledger.failed_calls,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise BackendError("TDSE ledger counts must be nonnegative integers")
        if ledger.limit is not None and (
            type(ledger.limit) is not int
            or ledger.limit < 1
            or ledger.attempted + ledger.reserved > ledger.limit
        ):
            raise BackendError("saved TDSE ledger exceeds its original call limit")
        if (
            ledger.diagnostic_calls > ledger.attempted
            or ledger.failed_calls > ledger.attempted
        ):
            raise BackendError("TDSE ledger subcounts exceed attempted calls")
        if self.limit is not None and ledger.attempted > self.limit:
            raise BackendError(
                "new TDSE call limit is below the cumulative attempted count"
            )
        self._attempted = ledger.attempted
        self._logical_evaluations = ledger.logical_evaluations
        self._diagnostic_calls = ledger.diagnostic_calls
        self._failed_calls = ledger.failed_calls
        self._last_failed = True
        self.release_reservation()

    def snapshot(self) -> AtomicCallLedger:
        """Return an immutable copy of the current accounting state.

        Returns
        -------
        AtomicCallLedger
            Counts including all successful, diagnostic and failed attempts.
        """
        return AtomicCallLedger(
            self.limit,
            self._attempted,
            self._reserved,
            self._logical_evaluations,
            self._diagnostic_calls,
            self._failed_calls,
        )


@runtime_checkable
class TDSEBinding(Protocol):
    """Provide the driver's explicit, resource-owning host atomic service."""

    @property
    def ledger(self) -> AtomicCallLedger:
        """Return the current immutable atomic-call accounting snapshot.

        Returns
        -------
        AtomicCallLedger
            Binding-local budget and attempted-call counts.
        """
        ...

    def evaluate(
        self,
        fields: Float[np.ndarray, "n_nodes n_to"],
        *,
        z: float,
        purpose: TDSEPurpose,
        density: float | None = None,
    ) -> TDSEResult:
        """Evaluate all radial rows using the binding's stable provider chunks.

        Parameters
        ----------
        fields : numpy.ndarray
            Real host fields in V/m, shape ``(n_nodes, n_to)``.
        z : float
            Optical position in m for provenance and failures.
        purpose : {'startup', 'predictor', 'corrected', 'diagnostic'}
            Label of the evaluated optical field.
        density : float or None, optional
            Number density at `z` in m^-3, a provenance label only (the source
            applies the density itself); ``None`` when the caller has none.

        Returns
        -------
        TDSEResult
            Validated owning microscopic arrays with no density or calibration.
        """
        ...

    def reserve(self, logical_calls: int) -> None:
        """Reserve every atomic chunk needed by an upcoming optical trial.

        Each reservation opens a new optical trial: its evaluations carry the
        trial's label until the next reservation (SEP 20 section 5.8).

        Parameters
        ----------
        logical_calls : int
            Positive maximum number of all-node source evaluations.
        """
        ...

    def record_trial(self, *, accepted: bool) -> None:
        """Record the stepper's decision on the most recently reserved trial.

        A provenance event only: it changes no accounting and no atomic state. The
        stepper reports every attempt, including one rejected before any call.

        Parameters
        ----------
        accepted : bool
            Whether the trial's step was accepted.
        """
        ...

    def release_reservation(self) -> None:
        """Release unused calls while retaining every attempted-call charge."""
        ...

    def restore_ledger(self, ledger: AtomicCallLedger) -> None:
        """Restore cumulative attempted calls before a same-model resumed run.

        Parameters
        ----------
        ledger : AtomicCallLedger
            Previous checkpoint's accounting; unused reservations are released.
        """
        ...

    def metadata(self) -> Mapping[str, object]:
        """Return serializable runtime, scientific and call-accounting records.

        Returns
        -------
        Mapping[str, object]
            Public provider configuration and actual execution provenance.
        """
        ...

    def memory_ledger(self) -> Mapping[str, int]:
        """Return named persistent allocations in bytes without native handles.

        Returns
        -------
        Mapping[str, int]
            Allocation names and byte counts.
        """
        ...

    def close(self) -> None:
        """Close every provider resource owned by the binding."""
        ...

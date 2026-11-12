# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run Pyroraptor inside the source worker: the TDSE binding's provider adapter.

The one place solprop imports the optional provider, and only in the worker
(:mod:`solprop.driver._source_worker`, SEP 49 section 6.5). It builds the provider's
models and plans from the plain records the binding sends, makes its atomic calls and
returns their results as plain records, with the provider's errors classified. Budgets,
the archive, validation, normalisation and warnings stay in the binding
(:mod:`solprop.driver.tdse`), in solprop's own process.

Requests: ``basis_from_preset``, ``hacc_preset`` and ``hacc_basis`` (queries); then for
a run ``prepare``, ``build_plans`` and ``evaluate``; for a consistency check
``diagnostic_plan``, ``diagnostic_evaluate`` and ``diagnostic_close``.
"""

import importlib
import json
import sys
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any, Literal, Protocol, cast, runtime_checkable

import numpy as np

from solprop.driver._source_worker import Record
from solprop.exceptions import BackendError, DriverError, MaterialError, SolpropError
from solprop.nonlinear.tdse import HaccBasis, TDSEBasis
from solprop.translate.tdse import HACC_NORM_GROWTH_LIMIT

__all__ = ["PyroraptorAdapter", "make_adapter"]

# Native classes cannot occur in runtime annotations: optional-package-free test
# collection resolves every annotation through beartype. Confine their dynamic
# typing to this adapter; the seam's scientific records and arrays remain typed.
_Native = Any
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
# The shadow components the binding reads, computed here where the record's own
# ``component`` method is (section 6.5, plain-data results).
_COMPONENTS = ("total_current", "current", "charge_dipole")
_PRESET_FIELDS = (
    "identity",
    "variant",
    "step_margin",
    "accuracy",
    "box_sensitivity",
    "gauge_sensitivity",
    "max_peak_intensity",
    "wavelength",
    "window",
    "device_admissible",
    "source",
)


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
    solve_preset: Callable[[str, str], _Native]
    unit_scale: Callable[[str], _Native]


def _load_provider() -> _Provider:
    """Load the optional package, in the worker only."""
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


def _identity(provider: _Provider) -> dict[str, object]:
    """Return the provider's build and installed distribution, for the handshake."""
    record: dict[str, object] = {"native": asdict(provider.build_info())}
    try:
        package = distribution("pyroraptor")
    except PackageNotFoundError:
        record["distribution"] = None
    else:
        record["distribution"] = package.version
        source = package.read_text("direct_url.json")
        record["direct_url"] = None if source is None else json.loads(source)
    return record


def _admit_memory(ledger: Mapping[str, int], limit: int | None) -> None:
    """Apply the declared atomic byte limit.

    - ``ledger``: named persistent allocations or estimates in bytes.
    - ``limit``: positive byte ceiling, or None for an unrestricted allocation.
    """
    total = sum(ledger.values())
    if limit is not None and total > limit:
        raise DriverError(
            f"TDSE persistent data requires {total} bytes, exceeding {limit} bytes"
        )


def _is(value: object, kind: type) -> bool:
    """Whether a provider object, or a record sent of one, is of `kind`."""
    if isinstance(value, Record):
        return value.record_type == f"{kind.__module__}.{kind.__qualname__}"
    return isinstance(value, kind)


def _hacc_surface(provider: _Provider, revision: str) -> _Native:
    """Return the provider's haCC namespace, refusing a build without it."""
    hacc = getattr(provider, "hacc", None)
    missing = [name for name in _HACC_SURFACE if not hasattr(hacc, name)]
    if missing:
        raise BackendError(
            "the installed TDSE provider has no haCC consumer surface "
            f"(missing {', '.join(missing)}); install Pyroraptor at the pinned "
            f"revision {revision}"
        )
    return hacc


def _hacc_fields(  # noqa: PLR0917 - called positionally
    hacc: _Native,
    atom: _Native,
    basis: _Native,
    gauge: _Native,
    cut_energy: float,
    max_step_au: float,
) -> dict[str, object]:
    """Convert provider haCC objects into the numeric fields of a ``HaccBasis``.

    - ``atom``, ``basis``, ``gauge``: provider objects, or the records of them a caller
      sent.
    - ``cut_energy``: spectral cut in J.
    - ``max_step_au``: RK4 step ceiling in atomic units of time.
    - Returns: every ``HaccBasis`` field but ``source`` and ``preset``.
    """
    if _is(atom, hacc.Argon):
        raise MaterialError(
            "argon TDCIS is not representable as a HaccBasis: the haCC target is "
            "helium only"
        )
    if not _is(atom, hacc.Helium):
        raise MaterialError("a HaccBasis needs a provider hacc.Helium atom")
    if atom.ions not in (1, 2):
        raise MaterialError(
            f"Helium(ions={atom.ions}) is not representable: HaccBasis admits "
            "haCC(1) (ions=1) and haCC(5) (ions=2)"
        )
    if _is(gauge, hacc.MixedGauge):
        gauge_radius: float | None = float(gauge.radius)
    elif _is(gauge, hacc.VelocityGauge):
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
    """Return a ``HaccBasis``'s fields without its provenance and preset identity."""
    record = asdict(basis)
    del record["source"], record["preset"]
    return record


def _hacc_objects(
    provider: _Provider, hacc: _Native, basis: HaccBasis
) -> tuple[_Native, _Native, _Native]:
    """Build the provider's atom, basis and gauge from a ``HaccBasis``."""
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


def check_info(
    info: _Native,
    *,
    rows: int,
    samples: int,
    spacing: float,
    outputs: str | None,
    cut_energy: float,
    closure: bool,
    preset: bool,
    model_match: bool,
) -> None:
    """Require a haCC plan's or result's ``info`` to describe exactly its model.

    - ``model_match``: whether ``info``'s atom, basis and gauge equal the model's, as
      the provider compares them (in the worker).
    - ``preset``: whether the model is a qualified preset, which must close.
    """
    if (
        info.rows != rows
        or info.samples != samples
        or info.spacing != spacing
        or info.shadow_order != 1
        or info.outputs != outputs
        or not model_match
        or info.cut_energy != cut_energy
    ):
        raise BackendError(
            "haCC plan metadata disagree with the prepared model or contract"
        )
    if bool(info.total_current_qualified) != closure:
        if preset:
            raise BackendError(
                "provider contract violation: a haCC preset reports no "
                "escaped-electron closure"
            )
        raise BackendError("haCC plan closure flag disagrees with its output mode")


def _rebuild[T](kind: type[T], record: object) -> T:
    """Return a solprop dataclass from the record of it the binding sent."""
    values = vars(record) if isinstance(record, Record) else cast("dict", record)
    return kind(**{item.name: values[item.name] for item in fields(kind) if item.init})  # pyright: ignore[reportArgumentType]


@dataclass
class _HaccModel:
    """One prepared haCC model and the output mode its plans use (SEP 20, C1).

    The model is entered into the adapter's stack first, so it closes after every plan
    built on it.
    """

    hacc: _Native
    basis: HaccBasis
    atom: _Native
    native_basis: _Native
    gauge: _Native
    preset: _Native
    model: _Native
    outputs: Literal["full", "total_current"] | None
    closure: bool | None

    def plan(self, *, rows: int, spacing: float, samples: int) -> _Native:
        """Construct one response plan, choosing the native output mode once.

        A custom mixed-gauge basis is first asked for the compact closed current; the
        provider refuses it with ``ValueError`` when the gauge radius admits no closure,
        and the full record is then used, as for the velocity gauge. For a full archive
        every plan is full, and a custom mixed-gauge basis takes its closure from the
        first plan's ``total_current_qualified``.
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
                self.closure = False
            else:
                self.outputs = "total_current"
                self.closure = True
                return plan
        plan = self.hacc.ResponsePlan(self.model, outputs=self.outputs, **arguments)
        if self.closure is None:
            self.closure = bool(plan.info.total_current_qualified)
        return plan

    def matches(self, info: _Native) -> bool:
        """Whether ``info`` names this model's atom, basis and gauge."""
        return bool(
            info.atom == self.atom
            and info.basis == self.native_basis
            and info.gauge == self.gauge
        )

    def check(self, plan: _Native, *, rows: int, samples: int, spacing: float) -> None:
        """Check a plan this model built."""
        check_info(
            plan.info,
            rows=rows,
            samples=samples,
            spacing=spacing,
            outputs=self.outputs,
            cut_energy=self.basis.cut_energy,
            closure=self.closure is True,
            preset=self.preset is not None,
            model_match=self.matches(plan.info),
        )

    def record(self) -> dict[str, object]:
        """Return what the binding needs of the model: mode, closure, preset."""
        preset = self.preset
        return {
            "outputs": self.outputs,
            "closure": self.closure,
            "preset": None
            if preset is None
            else {name: getattr(preset, name) for name in _PRESET_FIELDS},
        }


class PyroraptorAdapter:
    """Serve the TDSE binding's requests with Pyroraptor, inside the worker.

    - ``provider``: the provider module; None loads ``pyroraptor`` (the manufactured
      provider of the tests is passed here).
    """

    def __init__(self, provider: _Provider | None = None) -> None:
        self._provider = _load_provider() if provider is None else provider
        self._stack = ExitStack()
        self._plans: dict[int, _Native] = {}
        self._hacc: _HaccModel | None = None
        self._request: Mapping[str, Any] = {}
        self._diagnostic: tuple[ExitStack, _Native] | None = None

    # -- the worker's protocol ------------------------------------------------------

    def identity(self) -> Mapping[str, object]:
        """Return the provider's build record."""
        return _identity(self._provider)

    def classify(self, error: BaseException) -> str:
        """Classify a provider-side exception for the failure record (SEP 20, 5.6).

        ``native`` for a provider ``DrivenError``, ``contract`` for solprop's own
        refusal, ``interface`` for a ``ValueError`` or ``TypeError``, ``unexpected``
        otherwise.
        """
        if isinstance(error, self._provider.DrivenError):
            return "native"
        if isinstance(error, SolpropError):
            return "contract"
        if isinstance(error, ValueError | TypeError):
            return "interface"
        return "unexpected"

    def computed(self, value: object) -> Mapping[str, object]:
        """Return a row outcome's ``ok`` and a shadow's components, beside fields."""
        if hasattr(value, "failure") and hasattr(value, "ok"):
            return {"ok": bool(cast("Any", value).ok)}
        component = getattr(value, "component", None)
        orders = getattr(value, "orders", None)
        if not callable(component) or not isinstance(orders, tuple):
            return {}
        found: dict[str, object] = {}
        for quantity in _COMPONENTS:
            for order in cast("tuple[int, ...]", orders):
                try:
                    found[f"{quantity}:{order}"] = np.asarray(
                        component(quantity, order)
                    )
                except Exception as error:
                    found[f"{quantity}:{order}"] = {
                        "error": str(error),
                        "type": type(error).__name__,
                    }
        return {"components": found}

    def handle(self, kind: str, value: object) -> object:
        """Serve one request of the binding."""
        request = cast("Mapping[str, Any]", value)
        handlers: dict[str, Callable[[Mapping[str, Any]], object]] = {
            "basis_from_preset": self._basis_from_preset,
            "hacc_preset": self._hacc_preset,
            "hacc_basis": self._hacc_basis,
            "prepare": self._prepare,
            "build_plans": self._build_plans,
            "evaluate": self._evaluate,
            "diagnostic_plan": self._diagnostic_plan,
            "diagnostic_evaluate": self._diagnostic_evaluate,
            "diagnostic_close": self._diagnostic_close,
        }
        if kind not in handlers:
            raise BackendError(f"the TDSE provider adapter has no request {kind!r}")
        return handlers[kind](request)

    def close(self) -> None:
        """Close any diagnostic plan, then the plans before the setup they use."""
        self._diagnostic_close({})
        self._stack.close()

    # -- queries --------------------------------------------------------------------

    def _basis_from_preset(self, request: Mapping[str, Any]) -> object:
        """Return a provider preset's SAE basis fields, provenance apart."""
        selected = self._provider.solve_preset(request["target"], request["preset"])
        basis = selected.discretization
        scaling = basis.scaling
        return {
            "name": selected.name,
            "radial_edges": basis.radial.edges,
            "radial_orders": basis.radial.orders,
            "lmax": basis.angular.lmax,
            "cut_energy": selected.cut_energy,
            "provider_source": selected.source,
            "m": basis.angular.m,
            "potential": basis.model.name,
            "potential_version": basis.model.version,
            "scaling_radius": None if scaling is None else scaling.radius,
            "scaling_angle": None if scaling is None else scaling.angle,
            "infinite_order": None if scaling is None else scaling.infinite_order,
            "decay": None if scaling is None else scaling.decay,
            "cuts": basis.cuts,
        }

    def _hacc_preset(self, request: Mapping[str, Any]) -> object:
        """Return a qualified haCC preset's basis fields and its records."""
        hacc = _hacc_surface(self._provider, request["revision"])
        preset = hacc.helium_preset(request["ions"], request["variant"])
        return {
            "fields": _hacc_fields(
                hacc,
                preset.atom,
                preset.basis,
                preset.gauge,
                preset.cut_energy,
                preset.max_step_au,
            ),
            "preset": {name: getattr(preset, name) for name in _PRESET_FIELDS},
        }

    def _hacc_basis(self, request: Mapping[str, Any]) -> object:
        """Return the basis fields of a caller's provider objects, sent as records."""
        hacc = _hacc_surface(self._provider, request["revision"])
        return _hacc_fields(
            hacc,
            request["atom"],
            request["basis"],
            request["gauge"],
            request["cut_energy"],
            request["max_step_au"],
        )

    # -- a run ----------------------------------------------------------------------

    def _prepare(self, request: Mapping[str, Any]) -> object:
        """Prepare the model and, unless deferred, its plans; admit the memory.

        - ``request``: the descriptor's plain records, the chunk sizes, the memory the
          binding already holds and its limit.
        """
        self._request = request
        memory = dict(request["memory"])
        if request["model"] == "hacc":
            model = self._prepare_hacc(_rebuild(HaccBasis, request["basis"]))
            memory.update(self._model_memory(model))
            # Admit the model's measured ledger before any plan is built (C7).
            _admit_memory(memory, request["memory_limit"])
            if not request["defer_plans"]:
                memory.update(self._build_hacc_plans())
        else:
            memory.update(self._prepare_sae(_rebuild(TDSEBasis, request["basis"])))
        _admit_memory(memory, request["memory_limit"])
        return self._prepared(memory)

    def _prepared(self, memory: Mapping[str, int]) -> dict[str, object]:
        """Return the measured memory, the plan records and the haCC model's record."""
        return {
            "memory": dict(memory),
            "plans": {
                str(rows): asdict(plan.info) for rows, plan in self._plans.items()
            },
            "hacc": None if self._hacc is None else self._hacc.record(),
        }

    def _prepare_sae(self, basis: TDSEBasis) -> dict[str, int]:
        """Allocate shared SAE resources and one plan per distinct batch size."""
        provider, request = self._provider, self._request
        model = provider.atomic_model(basis.potential)
        if model.version != basis.potential_version:
            raise BackendError(
                "TDSE atomic potential revision differs from its descriptor"
            )
        scaling = None
        if basis.scaling_radius is not None:
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
        initial = provider.StateSelection(*request["initial"])
        setup = self._stack.enter_context(
            provider.SpectralSetup(
                operators,
                provider.SpectralOptions(
                    cut_energy=basis.cut_energy,
                    states=(initial,),
                    growing=request["growing"],
                    growth_tolerance=request["growth_tolerance"],
                    window_duration=request["window_duration"],
                    accept_removed_component=request["accept_removed_component"],
                ),
            )
        )
        memory: dict[str, int] = {}
        for label, ledger in (
            ("operators", operators.memory_ledger()),
            ("setup", setup.memory_ledger()),
        ):
            for entry in ledger.entries:
                memory[f"{label}.{entry.name}"] = entry.bytes
        step = (
            request["max_atomic_step"] / provider.unit_scale("time").si_per_atomic_unit
        )
        for rows in request["rows"]:
            plan = self._stack.enter_context(
                provider.ResponsePlan(
                    setup,
                    rows=rows,
                    spacing=request["spacing"],
                    samples=request["samples"],
                    max_step_au=step,
                    initial=initial,
                    field="electric_field",
                    shadow_order=request["shadow_order"],
                    complement=request["complement"],
                    outputs=request["outputs"],
                )
            )
            self._plans[rows] = plan
            for entry in plan.memory_ledger().entries:
                if not entry.shared:
                    memory[f"plan.{rows}.{entry.name}"] = entry.bytes
        return memory

    def _prepare_hacc(self, basis: HaccBasis) -> _HaccModel:
        """Prepare the haCC model, verifying a preset against its record."""
        request = self._request
        if request["complement"] == "vendor_blas":
            raise BackendError(
                "haCC response plans have no complement option; use the default "
                "TDSEExecution(complement='deterministic'), recorded as not "
                "applicable"
            )
        hacc = _hacc_surface(self._provider, request["revision"])
        atom, native_basis, gauge = _hacc_objects(self._provider, hacc, basis)
        preset: _Native = None
        if basis.preset is not None:
            preset = self._verified_preset(hacc, basis)
            prepared = preset.prepare()
        else:
            prepared = hacc.prepare(
                atom, native_basis, gauge=gauge, cut_energy=basis.cut_energy
            )
        model = self._stack.enter_context(prepared)
        # The native output mode: compact with the escaped-electron closure, full
        # without it (the provider refuses compact order one then, C3), and full
        # throughout for a full archive. Whether the plans carry the closure is known
        # beforehand for a preset (always) and for the velocity gauge (never); a
        # custom mixed gauge learns it from its first plan.
        outputs: Literal["full", "total_current"] | None = (
            "total_current" if basis.preset is not None else None
        )
        closure: bool | None = True if basis.preset is not None else None
        if basis.gauge_radius is None:
            outputs, closure = "full", False
        if request["archive_outputs"] == "full":
            outputs = "full"
        self._hacc = _HaccModel(
            hacc, basis, atom, native_basis, gauge, preset, model, outputs, closure
        )
        return self._hacc

    def _verified_preset(self, hacc: _Native, basis: HaccBasis) -> _Native:
        """Refuse a preset the installed provider does not reproduce."""
        preset = hacc.helium_preset(basis.ions, basis.variant)
        if preset.identity != basis.preset:
            raise BackendError(
                f"the installed provider's haCC preset is {preset.identity!r}, not "
                f"the expected {basis.preset!r}; regenerate the record"
            )
        found = _hacc_fields(
            hacc,
            preset.atom,
            preset.basis,
            preset.gauge,
            preset.cut_energy,
            preset.max_step_au,
        )
        if found != _numeric_fields(basis):
            raise BackendError(
                f"the installed provider's preset {preset.identity!r} does not "
                "reproduce the record's fields"
            )
        if self._request["device"] != "cpu" and not preset.device_admissible:
            raise BackendError(
                f"the haCC preset {preset.identity!r} is not admitted on the "
                f"{self._request['device']} device route"
            )
        return preset

    @staticmethod
    def _model_memory(model: _HaccModel) -> dict[str, int]:
        """Return the prepared model's measured ledger."""
        return {
            f"model.{entry.name}": entry.bytes
            for entry in model.model.memory_ledger().entries
        }

    def _build_hacc_plans(self) -> dict[str, int]:
        """Build and check one haCC plan per chunk size; return their memory."""
        model, request = self._hacc, self._request
        assert model is not None
        spacing = float(request["spacing"])
        memory: dict[str, int] = {}
        for rows in request["rows"]:
            plan = self._stack.enter_context(
                model.plan(rows=rows, spacing=spacing, samples=request["samples"])
            )
            model.check(plan, rows=rows, samples=request["samples"], spacing=spacing)
            self._plans[rows] = plan
            for entry in plan.memory_ledger().entries:
                if not entry.shared:
                    memory[f"plan.{rows}.{entry.name}"] = entry.bytes
        return memory

    def _build_plans(self, request: Mapping[str, Any]) -> object:
        """Build a deferred haCC binding's plans and admit them with its memory."""
        memory = {**request["memory"], **self._build_hacc_plans()}
        _admit_memory(memory, request["memory_limit"])
        return self._prepared(memory)

    def _kind(self, raw: object) -> str:
        """Name a result's record type: ``model.grid_response`` or the compact one."""
        if self._hacc is not None:
            namespace, prefix = self._hacc.hacc, "hacc"
        else:
            namespace, prefix = self._provider, "sae"
        if isinstance(raw, namespace.TotalCurrentResponse):
            return f"{prefix}.total_current_response"
        if isinstance(raw, namespace.GridResponse):
            return f"{prefix}.grid_response"
        return f"other:{type(raw).__qualname__}"

    def _result(self, raw: _Native) -> dict[str, object]:
        """Return one atomic call's record as returned, its type and model match."""
        info = getattr(raw, "info", None)
        match = None
        if self._hacc is not None and info is not None:
            try:
                match = self._hacc.matches(info)
            except AttributeError, TypeError, ValueError:
                match = False
        return {"kind": self._kind(raw), "raw": raw, "model_match": match}

    def _evaluate(self, request: Mapping[str, Any]) -> object:
        """Make one atomic call from the prepared state on one chunk's drive."""
        return self._result(self._plans[request["rows"]].evaluate(request["drive"]))

    # -- the haCC consistency diagnostic --------------------------------------------

    def _diagnostic_plan(self, request: Mapping[str, Any]) -> object:
        """Build a temporary one-row plan on the protocol lattice; return its memory."""
        model = self._hacc
        if model is None:
            raise BackendError("the consistency diagnostic needs a haCC binding")
        self._diagnostic_close({})
        stack = ExitStack()
        try:
            plan = stack.enter_context(
                model.plan(
                    rows=1, spacing=request["spacing"], samples=request["samples"]
                )
            )
            model.check(
                plan, rows=1, samples=request["samples"], spacing=request["spacing"]
            )
        except BaseException:
            stack.close()
            raise
        self._diagnostic = (stack, plan)
        return {
            "memory": {
                f"diagnostic.{entry.name}": entry.bytes
                for entry in plan.memory_ledger().entries
                if not entry.shared
            },
            "hacc": model.record(),
        }

    def _diagnostic_evaluate(self, request: Mapping[str, Any]) -> object:
        """Make the diagnostic's one atomic call."""
        if self._diagnostic is None:
            raise BackendError("no consistency diagnostic plan is open")
        return self._result(self._diagnostic[1].evaluate(request["drive"]))

    def _diagnostic_close(self, request: Mapping[str, Any]) -> object:
        """Close the temporary plan, if one is open."""
        del request
        if self._diagnostic is not None:
            stack, self._diagnostic = self._diagnostic[0], None
            stack.close()
        return None


def make_adapter() -> PyroraptorAdapter:
    """Return the adapter the worker serves with (its entry point)."""
    return PyroraptorAdapter()

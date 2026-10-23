# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared haCC helium records for the SEP 20 tests, with no provider import.

The tiny custom basis is the provider's own consumer example (edges 0 to 12 bohr,
order 6, particle l of 2): unconverged and fast, used only for contract checks.
The preset records mirror the provider's two qualified presets at definition
version 1; their numbers are the provider's, in SI.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from solprop.constants import AU_ENERGY, AU_TIME, BOHR_RADIUS
from solprop.nonlinear.tdse import (
    HaccBasis,
    TDSEAtomicWindow,
    TDSECalibration,
    TDSEGasResponse,
    TDSESampling,
)

A0 = BOHR_RADIUS
"""Bohr radius in m (CODATA, the provider's value)."""

PRESET_IDENTITIES = {
    1: "hacc-helium-1-v1-28f0eae146b1d665",
    2: "hacc-helium-5-v1-5853d5d50c187aa9",
}
"""The provider's version-1 preset identities (SEP 20 section 5.7, C2)."""

FAST_PRESET_IDENTITIES = {
    1: "hacc-helium-1-fast-v1-e38292e1b30729c1",
    2: "hacc-helium-5-fast-v1-e741d3ce524adc7e",
}
"""The provider's version-1 fast-variant identities (cut 125 hartree, step 0.02045)."""

PRESET_NUMERICS = {"standard": (200.0, 0.0088, 0.62), "fast": (125.0, 0.02045, 0.9)}
"""Each variant's cut (hartree), RK4 step ceiling (au) and step margin."""

FAST_DELTA = 1.5e-4
"""The fast variant's addition to the standard records (Pyroraptor's fast preset)."""

PRESET_EDGES_BOHR = (0.0, 0.5, 1.0, 2.0, 4.0, *(4.0 * k for k in range(2, 21)))


def tiny_basis(**changes: object) -> HaccBasis:
    """Return the provider's tiny unconverged custom haCC(1) basis.

    Parameters
    ----------
    **changes : object
        Field replacements.

    Returns
    -------
    HaccBasis
        Custom record, ``preset=None``.
    """
    basis = HaccBasis(
        1,
        1,
        tuple(x * A0 for x in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0)),
        (6,) * 6,
        "particle",
        2,
        12 * A0,
        0.15,
        6,
        1.0 / A0,
        4 * A0,
        30 * AU_ENERGY,
        0.05,
        source="tiny unconverged custom basis for contract tests",
    )
    return dataclasses.replace(basis, **changes) if changes else basis


def preset_basis(ions: int = 1, variant: str = "standard") -> HaccBasis:
    """Return a record equal to the provider's qualified preset at version 1.

    Parameters
    ----------
    ions : int
        1 for haCC(1), 2 for haCC(5).
    variant : {'standard', 'fast'}
        The provider's numerics variant.

    Returns
    -------
    HaccBasis
        Preset record with the provider's identity.
    """
    return HaccBasis(
        ions,  # pyright: ignore[reportArgumentType] - the tests pass 1 or 2
        6,
        tuple(x * A0 for x in PRESET_EDGES_BOHR),
        (10,) * (len(PRESET_EDGES_BOHR) - 1),
        "particle",
        50,
        80.0 * A0,
        0.15,
        20,
        0.5 / A0,
        20.0 * A0,
        PRESET_NUMERICS[variant][0] * AU_ENERGY,
        PRESET_NUMERICS[variant][1],
        source="provider preset evidence (contract test copy)",
        preset=(PRESET_IDENTITIES if variant == "standard" else FAST_PRESET_IDENTITIES)[
            ions
        ],
    )


def calibration(basis: HaccBasis) -> TDSECalibration:
    """Return a research calibration record for a haCC basis.

    Parameters
    ----------
    basis : HaccBasis
        The basis the record documents.

    Returns
    -------
    TDSECalibration
        Research record with an explicit limitation and no uncertainty.
    """
    return TDSECalibration(
        f"sep20-test-{basis.identity[:12]}",
        "research",
        basis.identity,
        (62e-9, 1.6e-6),
        source="contract test",
        limitations=("contract test only; no atomic accuracy claim",),
    )


def hacc_response(basis: HaccBasis | None = None, **changes: object) -> TDSEGasResponse:
    """Return a haCC ``tdse_native`` response at scale one.

    Parameters
    ----------
    basis : HaccBasis or None
        The basis; the tiny custom basis by default.
    **changes : object
        Keyword replacements of the response, for example ``atomic_window``.

    Returns
    -------
    TDSEGasResponse
        Validated haCC response with ``TDSESampling(1)`` and the model's default
        ``gated_derivative`` source unless replaced.
    """
    basis = tiny_basis() if basis is None else basis
    values: dict[str, object] = {
        "target": "helium",
        "basis": basis,
        "max_atomic_step": basis.max_step_au * AU_TIME,
        "kerr_model": "tdse_native",
        "remainder_scale": 1.0,
        "calibration": calibration(basis),
        "sampling": TDSESampling(4),
        "model": "hacc",
        "atomic_window": TDSEAtomicWindow(duration=8e-15, taper=1e-15),
    }
    values.update(changes)
    return TDSEGasResponse(**values)  # pyright: ignore[reportArgumentType] - test keywords


# --- a fake haCC provider: doubles of the SEP 20 section 5.7 contract (C1 to C8) ------


class FakeDrivenError(Exception):
    """Stand-in for ``pyroraptor.DrivenError``."""


@dataclass(frozen=True)
class FakeBuild:
    execution_space: str = "OpenMP"
    source_revision: str = "876b72d"
    requested_threads: int = 1


@dataclass(frozen=True)
class FakeHelium:
    ions: int = 1
    neutral_square: int = 6


@dataclass(frozen=True)
class FakeArgon:
    thresholds: str = "experimental_holes"


@dataclass(frozen=True)
class FakeAngularLimit:
    particle: int | None = None
    total: int | None = None


@dataclass(frozen=True)
class FakeRadialBasis:
    edges: tuple[float, ...]
    orders: tuple[int, ...]


@dataclass(frozen=True)
class FakeScaling:
    radius: float
    angle: float
    infinite_order: int
    decay: float


@dataclass(frozen=True)
class FakeBasis:
    radial: FakeRadialBasis
    scaling: FakeScaling
    angular: FakeAngularLimit


@dataclass(frozen=True)
class FakeMixedGauge:
    radius: float


@dataclass(frozen=True)
class FakeVelocityGauge:
    pass


@dataclass(frozen=True)
class FakeInfo:
    rows: int
    samples: int
    spacing: float
    route: str
    substeps: int
    step_au: float
    step: float
    steps: int
    stepper: str
    radius: float
    gauge_radius: float | None
    cut_energy: float
    atom: object
    basis: object
    gauge: object
    dimension: int
    monitor_interval: int
    norm_growth_limit: float
    total_current_qualified: bool
    build: FakeBuild
    shadow_order: int = 0
    outputs: str = "full"
    closure_radius: float | None = None


@dataclass(frozen=True)
class FakeOutcome:
    failure: str = "none"
    failure_step: int | None = None

    @property
    def ok(self):
        return self.failure == "none"


@dataclass(frozen=True, eq=False)
class FakeMonitors:
    initial_norm: np.ndarray
    norm: np.ndarray
    peak_norm: np.ndarray
    monitored_step: np.ndarray
    non_finite: np.ndarray


@dataclass(frozen=True, eq=False)
class FakeShadow:
    orders: tuple[int, ...]
    charge_dipole: np.ndarray
    current: np.ndarray
    total_current: np.ndarray | None = None

    def component(self, quantity, order):
        if quantity == "total_current":
            if self.total_current is None:
                raise ValueError("total_current has no orders without the closure")
            return self.total_current[:, 0, :]
        return getattr(self, quantity)[:, self.orders.index(order), :]


class FakeTotalCurrentResponse(SimpleNamespace):
    pass


class FakeGridResponse(SimpleNamespace):
    pass


def manufactured_currents(fields, spacing, law="pointwise"):
    """Return a manufactured (j1, j_total) with an exact order-one part.

    ``j1 = 2^-3 E``; the pointwise law adds ``2^-7 E^3 + 2^-11 E^5``; the causal law
    adds a running sum of ``2^-7 E^3``, which depends on the drive up to each time
    only. Dyadic coefficients keep the algebra exact.
    """
    first = 2.0**-3 * fields
    if law == "causal":
        rest = np.cumsum(2.0**-7 * fields**3, axis=1)
    else:
        rest = 2.0**-7 * fields**3 + 2.0**-11 * fields**5
    return first, first + rest


class FakeHaccModel:
    def __init__(self, fake, atom, basis, gauge, cut_energy):
        self.fake = fake
        self.atom = atom
        self.basis = basis
        self.gauge = gauge
        self.cut_energy = float(cut_energy)
        fake.prepared.append((atom, basis, gauge, cut_energy))

    def memory_ledger(self):
        return SimpleNamespace(
            entries=(
                SimpleNamespace(
                    name="record",
                    bytes=self.fake.model_bytes,
                    memory_space="Host",
                    shared=True,
                ),
            )
        )

    def __enter__(self):
        self.fake.events.append("enter-model")
        return self

    def __exit__(self, *args):
        self.fake.events.append("close-model")


class FakeHaccPlan:
    def __init__(self, fake, model, **values):
        fake.plan_arguments.append(dict(values))
        if fake.failure == "plan_driven" and values["rows"] == fake.fail_rows:
            raise FakeDrivenError("synthetic plan allocation failure")
        closure = fake.closure_for(model)
        if (
            values["outputs"] == "total_current"
            and values["shadow_order"] == 1
            and not closure
        ):
            raise ValueError(
                "outputs='total_current' with shadow_order=1 needs the "
                "escaped-electron closure (helium with a mixed gauge)"
            )
        self.fake = fake
        self.model = model
        self.closure = closure
        gauge = model.gauge
        self.info = FakeInfo(
            rows=values["rows"],
            samples=values["samples"],
            spacing=values["spacing"],
            route=values["field"],
            substeps=1,
            step_au=values["max_step_au"],
            step=values["spacing"],
            steps=values["samples"] - 1,
            stepper="rk4",
            radius=model.basis.scaling.radius,
            gauge_radius=getattr(gauge, "radius", None),
            cut_energy=model.cut_energy,
            atom=model.atom,
            basis=model.basis,
            gauge=gauge,
            dimension=10,
            monitor_interval=1,
            norm_growth_limit=values["norm_growth_limit"],
            total_current_qualified=closure,
            build=fake.build,
            shadow_order=values["shadow_order"],
            outputs=values["outputs"],
            closure_radius=1e-10 if closure else None,
        )
        if fake.failure == "plan_info":
            self.info = dataclasses.replace(self.info, atom=FakeHelium(2, 6))
        elif fake.failure == "plan_unqualified":
            self.info = dataclasses.replace(self.info, total_current_qualified=False)
        self.name = f"plan-{values['rows']}-{values['samples']}"

    def memory_ledger(self):
        return SimpleNamespace(
            entries=(
                SimpleNamespace(
                    name="model.record",
                    bytes=self.fake.model_bytes,
                    memory_space="Host",
                    shared=True,
                ),
                SimpleNamespace(
                    name="march.state",
                    bytes=self.info.rows * self.fake.plan_bytes,
                    memory_space="Host",
                    shared=False,
                ),
            )
        )

    def __enter__(self):
        self.fake.events.append(f"enter-{self.name}")
        return self

    def __exit__(self, *args):
        self.fake.events.append(f"close-{self.name}")

    def evaluate(self, fields):
        fake = self.fake
        fake.calls.append(np.array(fields, copy=True))
        failure = fake.failure
        if failure in ("raise_driven", "raise_value", "raise_type", "work_then_raise"):
            if failure == "work_then_raise":
                fake.work_done += 1
            raise {
                "raise_driven": FakeDrivenError,
                "raise_value": ValueError,
                "raise_type": TypeError,
                "work_then_raise": FakeDrivenError,
            }[failure]("synthetic provider failure")
        first, total = manufactured_currents(fields, self.info.spacing, fake.law)
        if fake.response is not None:
            first, total = fake.response(fields, self.info.spacing)
        rows, samples = fields.shape
        peak = np.max(np.abs(total), axis=1, keepdims=True)
        bound = np.broadcast_to(fake.bound_scale * peak, (rows, samples)).copy()
        monitors = FakeMonitors(
            np.ones(rows),
            np.ones(rows),
            np.ones(rows),
            np.full(rows, samples - 1, dtype=np.int64),
            np.zeros(rows, dtype=bool),
        )
        common = {
            "time": np.arange(samples) * self.info.spacing,
            "total_current": total,
            "neglected_current_bound": bound if self.closure else None,
            "monitors": monitors,
            "outcomes": tuple(FakeOutcome() for _ in range(rows)),
            "valid": np.ones((rows, samples), dtype=bool),
            "info": self.info,
        }
        if self.info.outputs == "total_current":
            raw = FakeTotalCurrentResponse(
                **common,
                shadow_total_current=first[:, None, :].copy(),
                shadow_orders=(1,),
                radius=self.info.radius,
            )
        else:
            raw = FakeGridResponse(
                **common,
                # The full record's remaining observables (C4), manufactured.
                vector_potential=-np.cumsum(fields, axis=1) * self.info.spacing,
                electric_field=fields.copy(),
                outside=np.zeros_like(first),
                current=first,
                charge_dipole=np.zeros_like(first),
                shadow=FakeShadow(
                    (1,),
                    np.zeros_like(first)[:, None, :],
                    first[:, None, :].copy(),
                    first[:, None, :].copy() if self.closure else None,
                ),
            )
        raw = inject(raw, failure)
        fake.responses.append(raw)
        return raw


def inject(raw, failure):  # noqa: PLR0912 - one injected malformation per branch
    """Apply one named malformation to a fake result and return it."""
    if failure is None:
        return raw
    rows = raw.total_current.shape[0]
    monitors = raw.monitors
    if failure == "row":
        raw.outcomes = (FakeOutcome("norm_growth", 7), *raw.outcomes[1:])
    elif failure == "valid":
        raw.valid[0, -1] = False
    elif failure == "labels":
        raw.shadow_orders = (1, 3)
    elif failure == "bound_none":
        raw.neglected_current_bound = None
    elif failure == "bound_negative":
        raw.neglected_current_bound[0, 0] = -1.0
    elif failure == "bound_nonfinite":
        raw.neglected_current_bound[0, 0] = np.inf
    elif failure == "bound_shape":
        raw.neglected_current_bound = raw.neglected_current_bound[:, :-1]
    elif failure == "info":
        raw.info = dataclasses.replace(raw.info, spacing=raw.info.spacing * 2)
    elif failure == "info_atom":
        raw.info = dataclasses.replace(raw.info, atom=FakeHelium(2, 6))
    elif failure == "wrong_type":
        return SimpleNamespace(**vars(raw))
    elif failure == "extents":
        raw.shadow_total_current = raw.shadow_total_current[:, :, :-1]
    elif failure == "current_extents":
        raw.total_current = raw.total_current[:, :-1]
    elif failure.startswith("monitor_dtype_"):
        name = failure.removeprefix("monitor_dtype_")
        wrong = np.zeros(rows, dtype=np.int32)
        raw.monitors = dataclasses.replace(monitors, **{name: wrong})
    elif failure == "monitor_shape":
        raw.monitors = dataclasses.replace(monitors, norm=np.ones(rows + 1))
    elif failure == "monitor_nonfinite":
        raw.monitors = dataclasses.replace(monitors, peak_norm=np.full(rows, np.nan))
    elif failure == "nonfinite":
        raw.total_current[0, -1] = np.nan
    elif failure == "full_shadow_missing":
        raw.shadow = None
    elif failure == "full_labels":
        raw.shadow = dataclasses.replace(raw.shadow, orders=(1, 3))
    return raw


@dataclass
class FakePreset:
    name: str
    variant: str
    step_margin: float
    identity: str
    atom: object
    basis: object
    gauge: object
    cut_energy: float
    max_step_au: float
    accuracy: float
    box_sensitivity: float
    gauge_sensitivity: float
    max_peak_intensity: float
    wavelength: float
    window: float
    inner_nodes: int
    device_admissible: bool
    source: str
    fake: object = None

    def prepare(self):
        return FakeHaccModel(
            self.fake, self.atom, self.basis, self.gauge, self.cut_energy
        )


def fake_objects(basis: HaccBasis):
    """Return the fake provider objects describing a ``HaccBasis``."""
    angular = (
        FakeAngularLimit(particle=basis.angular_order)
        if basis.angular_limit == "particle"
        else FakeAngularLimit(total=basis.angular_order)
    )
    native = FakeBasis(
        FakeRadialBasis(basis.radial_edges, basis.radial_orders),
        FakeScaling(
            basis.scaling_radius,
            basis.scaling_angle,
            basis.infinite_order,
            basis.decay,
        ),
        angular,
    )
    gauge = (
        FakeVelocityGauge()
        if basis.gauge_radius is None
        else FakeMixedGauge(basis.gauge_radius)
    )
    return FakeHelium(basis.ions, basis.neutral_square), native, gauge


class FakeHacc:
    """The ``pyroraptor.hacc`` namespace of the fake."""

    AngularLimit = FakeAngularLimit
    Argon = FakeArgon
    Basis = FakeBasis
    GridResponse = FakeGridResponse
    Helium = FakeHelium
    MixedGauge = FakeMixedGauge
    TotalCurrentResponse = FakeTotalCurrentResponse
    VelocityGauge = FakeVelocityGauge

    def __init__(self, fake):
        self.fake = fake

    def helium_preset(self, ions, variant="standard"):
        self.fake.preset_requests.append(ions)
        self.fake.variant_requests.append(variant)
        record = preset_basis(ions, variant)
        atom, native, gauge = fake_objects(record)
        delta = 0.0 if variant == "standard" else FAST_DELTA
        preset = FakePreset(
            name="haCC(1)" if ions == 1 else "haCC(5)",
            variant=variant,
            step_margin=PRESET_NUMERICS[variant][2],
            identity=(
                PRESET_IDENTITIES if variant == "standard" else FAST_PRESET_IDENTITIES
            )[ions],
            atom=atom,
            basis=native,
            gauge=gauge,
            cut_energy=record.cut_energy,
            max_step_au=record.max_step_au,
            accuracy=0.0122 + delta,
            box_sensitivity=(0.0095 if ions == 1 else 0.0093) + delta,
            gauge_sensitivity=0.0011 + delta,
            max_peak_intensity=8.0e18,
            wavelength=800e-9,
            window=self.fake.preset_window,
            inner_nodes=208,
            device_admissible=self.fake.device_admissible,
            source="synthetic preset evidence",
            fake=self.fake,
        )
        for name, value in self.fake.preset_changes.items():
            setattr(preset, name, value)
        return preset

    def prepare(self, atom, basis, *, gauge, cut_energy=None):
        return FakeHaccModel(self.fake, atom, basis, gauge, cut_energy)

    def ResponsePlan(self, model, **values):  # noqa: N802 - the provider's class name
        return FakeHaccPlan(self.fake, model, **values)


@dataclass
class FakeHaccProvider:
    """A fake provider with a haCC namespace and the SAE attributes it must carry."""

    build: FakeBuild = field(default_factory=FakeBuild)
    failure: str | None = None
    fail_rows: int = 1
    law: str = "pointwise"
    response: object = None
    bound_scale: float = 0.25
    closure_admitted: bool = True
    preset_window: float = 60e-15
    device_admissible: bool = True
    preset_changes: dict = field(default_factory=dict)
    model_bytes: int = 1000
    plan_bytes: int = 100
    events: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    responses: list = field(default_factory=list)
    prepared: list = field(default_factory=list)
    plan_arguments: list = field(default_factory=list)
    preset_requests: list = field(default_factory=list)
    variant_requests: list = field(default_factory=list)
    work_done: int = 0

    DrivenError = FakeDrivenError
    GridResponse = FakeGridResponse
    TotalCurrentResponse = FakeTotalCurrentResponse

    def __post_init__(self):
        self.hacc = FakeHacc(self)

    def closure_for(self, model):
        return isinstance(model.gauge, FakeMixedGauge) and self.closure_admitted

    def build_info(self):
        return self.build

    def unit_scale(self, quantity):
        return SimpleNamespace(si_per_atomic_unit=AU_TIME)

    def RadialBasis(self, edges, orders):  # noqa: N802 - the provider's class name
        return FakeRadialBasis(tuple(edges), tuple(orders))

    def ComplexScaling(self, radius, angle, infinite_order, decay):  # noqa: N802
        return FakeScaling(radius, angle, infinite_order, decay)

    # The SAE surface is present but unused by these tests.
    def _unused(self, *args, **kwargs):
        raise AssertionError("the SAE surface is not used by haCC tests")

    AngularBasis = _unused
    Discretization = _unused
    ResponsePlan = _unused
    SpectralOptions = _unused
    SpectralSetup = _unused
    StateSelection = _unused
    atomic_model = _unused
    solve_preset = _unused


def install(monkeypatch, fake=None):
    """Make ``solprop.driver.tdse`` load a fake provider; return it."""
    from solprop.driver import tdse as runtime  # noqa: PLC0415 - keep tests importable

    fake = FakeHaccProvider() if fake is None else fake
    monkeypatch.setattr(runtime, "_load_provider", lambda: fake)
    return fake


def helium_alpha(omega):
    """Return optmat helium's per-atom susceptibility in m^3, constant beyond support.

    The fake's manufactured responses need a value at every FFT bin; outside optmat's
    declared support the end value is held, which the band rule never compares.
    """
    from solprop.physdata import species  # noqa: PLC0415 - optional for most tests

    susceptibility = species("He").susceptibility
    shortest = susceptibility.wavelength_range[0]
    top = 2 * np.pi * 299792458.0 / shortest
    inside = np.clip(omega, 1e12, top * (1 - 1e-12))
    return np.asarray(susceptibility.susceptibility(inside), dtype=np.complex128)


def alpha_law(k1, *, phase=0.0, cubic=0.0, lambda5=0.0, leakage=0.0, gamma=None):
    """Return a fake response law whose first order has ``alpha_optmat / k1``.

    ``j1 = IFFT[i omega eps0 conj(alpha) e^{i phase} / k1 FFT[E]]`` is linear in E;
    the total adds an optional cubic ``eps0 (gamma / k3) d/dt E^3`` (given as
    ``cubic = 1 / k3`` times optmat's gamma), a lambda^5 term and a first-order leak.
    """
    from solprop.constants import EPS0  # noqa: PLC0415

    def law(fields, spacing):
        samples = fields.shape[-1]
        omega = 2 * np.pi * np.fft.rfftfreq(samples, spacing)
        alpha = np.conj(helium_alpha(omega)) * np.exp(1j * phase) / k1
        spectrum = 1j * omega * EPS0 * alpha * np.fft.rfft(fields, axis=-1)
        first = np.fft.irfft(spectrum, n=samples, axis=-1)
        total = first + leakage * first
        if cubic:
            coefficient = cubic * (5.21e-53 if gamma is None else gamma)
            cube = np.fft.rfft(fields**3, axis=-1)
            total = total + np.fft.irfft(
                1j * omega * EPS0 * coefficient * cube, n=samples, axis=-1
            )
        if lambda5:
            total = total + lambda5 * fields**5
        return first, total

    return law


def plain(value: object) -> Any:
    """Return a JSON-compatible record as plain, untyped containers for assertions."""
    return json.loads(json.dumps(value, allow_nan=False))

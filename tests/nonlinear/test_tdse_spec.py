# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate frozen TDSE specifications without importing the optional provider."""

import dataclasses
import inspect
import json
import math
import subprocess
import sys

import pytest
from jaxtyping import TypeCheckError
from tests import hacc_cases
from tests.hacc_cases import A0, hacc_response, preset_basis, tiny_basis

from solprop.exceptions import MaterialError
from solprop.nonlinear import tdse
from solprop.nonlinear.responses import ResponseTraits
from solprop.nonlinear.tdse import (
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)


@pytest.fixture
def basis():
    return TDSEBasis(
        "manufactured",
        (0.0, 1e-9, 2e-9),
        (4, 6),
        5,
        1e-16,
        source="manufactured contract fixture; no atomic accuracy claim",
    )


@pytest.fixture
def calibration(basis):
    return TDSECalibration(
        "manufactured-research",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="manufactured test record",
        limitations=("not an atomic calibration",),
    )


@pytest.fixture
def response(basis, calibration):
    return TDSEGasResponse(
        "helium", basis, 1e-19, "optmat_cubic", 2.0, calibration, TDSESampling(4)
    )


def test_specs_do_not_import_optional_provider():
    code = (
        "import sys; import solprop.nonlinear.tdse; import solprop.translate.tdse; "
        "assert not any(name == 'pyroraptor' or name.startswith('pyroraptor.') "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_basis_is_resolved_serializable_and_immutable(basis):
    restored = json.loads(json.dumps(dataclasses.asdict(basis), allow_nan=False))
    assert restored["radial_edges"] == list(basis.radial_edges)
    assert restored["cut_energy"] == basis.cut_energy
    assert dataclasses.replace(basis).identity == basis.identity
    assert dataclasses.replace(basis, cut_energy=2e-16).identity != basis.identity
    with pytest.raises(dataclasses.FrozenInstanceError):
        basis.name = "changed"


def test_scaled_basis_requires_complete_public_discretization(basis):
    scaled = dataclasses.replace(
        basis,
        scaling_radius=2e-9,
        scaling_angle=0.15,
        infinite_order=12,
        decay=2e10,
        cuts=(1e-9,),
    )
    assert scaled.identity != basis.identity
    with pytest.raises(MaterialError, match="four scaling"):
        dataclasses.replace(basis, scaling_radius=2e-9)
    with pytest.raises(MaterialError, match="exactly"):
        dataclasses.replace(scaled, scaling_radius=1.5e-9)
    with pytest.raises(MaterialError, match="angle"):
        dataclasses.replace(scaled, scaling_angle=2.0)
    with pytest.raises(MaterialError, match="cuts"):
        dataclasses.replace(scaled, cuts=(2e-9,))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"radial_edges": (0.0, 1e-9)}, "one more"),
        ({"radial_edges": (1e-10, 1e-9, 2e-9)}, "first"),
        ({"radial_edges": (0.0, 2e-9, 1e-9)}, "increasing"),
        ({"radial_edges": (0.0, float("nan"), 2e-9)}, "finite"),
        ({"radial_orders": (1, 6)}, "integer"),
        ({"radial_orders": (65, 6)}, "64"),
        ({"lmax": -1}, "lmax"),
        ({"lmax": 4097}, "4096"),
        ({"m": 1}, "m=0"),
        ({"cut_energy": 0.0}, "positive"),
        ({"potential": "argon_tong05"}, "helium"),
        ({"source": " "}, "source"),
    ],
)
def test_basis_refuses_invalid_or_unsupported_configuration(basis, changes, message):
    with pytest.raises(MaterialError, match=message):
        dataclasses.replace(basis, **changes)


def test_calibration_cannot_claim_qualification_without_uncertainty(calibration):
    with pytest.raises(MaterialError, match="uncertainty"):
        dataclasses.replace(calibration, status="qualified")
    qualified = dataclasses.replace(calibration, status="qualified", uncertainty=0.01)
    assert qualified.status == "qualified"
    with pytest.raises(MaterialError, match="limitations"):
        dataclasses.replace(calibration, limitations=())
    with pytest.raises(MaterialError, match="increasing"):
        dataclasses.replace(calibration, wavelength_range=(2e-6, 400e-9))
    with pytest.raises(MaterialError, match="uncertainty"):
        dataclasses.replace(calibration, uncertainty=float("nan"))


def test_closure_scales_are_independent_and_explicit(response):
    assert response.traits == ResponseTraits(
        "memory", ("field",), "scalar", ("density",)
    )
    shadow = dataclasses.replace(response, kerr_model="tdse_shadow", cubic_scale=1.537)
    assert shadow.remainder_scale == 2.0
    assert shadow.cubic_scale == 1.537
    native = dataclasses.replace(
        response, kerr_model="tdse_native", remainder_scale=1.0
    )
    assert native.cubic_scale is None
    with pytest.raises(MaterialError, match="requires cubic_scale"):
        dataclasses.replace(response, kerr_model="tdse_shadow")
    with pytest.raises(MaterialError, match="only for"):
        dataclasses.replace(response, cubic_scale=1.0)
    with pytest.raises(MaterialError, match="positive"):
        dataclasses.replace(shadow, cubic_scale=0.0)
    with pytest.raises(MaterialError, match="positive"):
        dataclasses.replace(response, max_atomic_step=float("inf"))
    with pytest.raises(MaterialError, match="helium"):
        dataclasses.replace(response, target="argon")


@pytest.mark.parametrize("scale", [1.0, 1.47, 2.0])
def test_native_nonlinear_scale_is_explicit_and_serialized(response, scale):
    native = dataclasses.replace(
        response, kerr_model="tdse_native", remainder_scale=scale
    )
    record = json.loads(json.dumps(dataclasses.asdict(native), allow_nan=False))
    assert native.remainder_scale == record["remainder_scale"] == scale
    assert native.kerr_model == record["kerr_model"] == "tdse_native"
    assert native.cubic_scale is record["cubic_scale"] is None
    assert native.calibration is response.calibration
    with pytest.raises(MaterialError, match="only for tdse_shadow"):
        dataclasses.replace(native, cubic_scale=1.0)


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, float("inf"), float("-inf"), float("nan"), True, False]
)
def test_native_nonlinear_scale_refuses_invalid_values(response, scale):
    with pytest.raises((MaterialError, TypeCheckError), match="remainder_scale"):
        dataclasses.replace(response, kerr_model="tdse_native", remainder_scale=scale)


def test_basis_and_calibration_identity_must_agree(response):
    different = dataclasses.replace(response.basis, cut_energy=2e-16)
    with pytest.raises(MaterialError, match="basis_identity"):
        dataclasses.replace(response, basis=different)
    too_small = dataclasses.replace(response.basis, lmax=1)
    calibration = dataclasses.replace(
        response.calibration, basis_identity=too_small.identity
    )
    with pytest.raises(MaterialError, match="lmax"):
        dataclasses.replace(response, basis=too_small, calibration=calibration)
    native = dataclasses.replace(
        response,
        basis=too_small,
        calibration=calibration,
        kerr_model="tdse_native",
        remainder_scale=1.0,
    )
    assert native.basis.lmax == 1


@pytest.mark.parametrize("factor", [0, -1, 3, 12, True])
def test_sampling_refuses_non_binary_refinement(factor):
    with pytest.raises((MaterialError, TypeCheckError), match="refinement"):
        TDSESampling(factor)


def test_execution_refuses_invalid_limits():
    for name in ("memory_limit", "max_batch_rows", "max_atomic_calls"):
        with pytest.raises(MaterialError, match=name):
            dataclasses.replace(TDSEExecution(), **{name: 0})
    assert TDSEExecution(device="hip", max_batch_rows=33).max_batch_rows == 33


@pytest.mark.parametrize("name", ["radial_edges", "radial_orders", "cuts"])
def test_basis_revalidation_refuses_mutable_sequences(basis, name):
    # Rehydrated records must retain immutability without the test-only type hook.
    record = dataclasses.replace(basis)
    object.__setattr__(record, name, list(getattr(record, name)))
    with pytest.raises(MaterialError, match="tuple"):
        record.__post_init__()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("status", "pending", "status"),
        ("wavelength_range", (400e-9,), "pair"),
        ("wavelength_range", [400e-9, 2e-6], "pair"),
    ],
)
def test_calibration_revalidation_refuses_unknown_status_or_domain(
    calibration, name, value, message
):
    record = dataclasses.replace(calibration)
    object.__setattr__(record, name, value)
    with pytest.raises(MaterialError, match=message):
        record.__post_init__()


def test_calibration_limitations_require_immutable_container_without_type_hook():
    # Unwrap test instrumentation to exercise the production tuple guard itself.
    with pytest.raises(MaterialError, match="limitations must be a tuple"):
        inspect.unwrap(tdse._strings)(["unqualified"], "limitations")


@pytest.mark.parametrize(
    ("name", "value"), [("device", "metal"), ("complement", "automatic")]
)
def test_execution_revalidation_refuses_unknown_runtime_policies(name, value):
    record = TDSEExecution()
    object.__setattr__(record, name, value)
    with pytest.raises(MaterialError, match=name):
        record.__post_init__()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("basis", {}, "basis and calibration record"),
        ("calibration", {}, "basis and calibration record"),
        ("sampling", {}, "sampling record"),
        ("kerr_model", "automatic", "Kerr model"),
    ],
)
def test_response_revalidation_refuses_unresolved_records_and_unknown_closure(
    response, name, value, message
):
    record = dataclasses.replace(response)
    object.__setattr__(record, name, value)
    with pytest.raises(MaterialError, match=message):
        record.__post_init__()


def test_the_response_has_no_source_filter_and_never_gates_a_cubic_closure(response):
    # SEP 26 and the removal of the option: the native source is always gated and
    # edge-free, the cubic replacements never; there is nothing left to choose.
    assert "source_filter" not in inspect.signature(TDSEGasResponse).parameters
    assert "source_filter" not in dataclasses.asdict(response)
    with pytest.raises(TypeError):
        TDSEGasResponse(
            **{
                name: getattr(response, name)
                for name in (
                    "target",
                    "basis",
                    "max_atomic_step",
                    "kerr_model",
                    "remainder_scale",
                    "calibration",
                    "sampling",
                )
            },
            source_filter="none",  # pyright: ignore[reportCallIssue] - the removed field
        )


@pytest.mark.parametrize("model", ["optmat_cubic", "tdse_shadow"])
def test_an_atomic_window_refuses_a_cubic_closure(response, model):
    # The window's gate is the native source's gate; a cubic source is never gated.
    with pytest.raises(MaterialError, match="requires kerr_model='tdse_native'"):
        dataclasses.replace(
            response,
            kerr_model=model,
            cubic_scale=1.537 if model == "tdse_shadow" else None,
            atomic_window=TDSEAtomicWindow(60e-15),
        )


# --- SEP 20: haCC records (section 7.1 items 1 to 3) -------------------------------


def _hacc():
    return tiny_basis()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("ions", 0, "ions"),
        ("ions", 3, "ions"),
        ("ions", True, "ions"),
        ("ions", 1.0, "ions"),
        ("neutral_square", -1, "neutral_square"),
        ("neutral_square", 8, "neutral_square"),
        ("neutral_square", True, "neutral_square"),
        ("radial_edges", [0.0, 1e-10, 2e-10], "tuples"),
        ("radial_orders", [6, 6], "tuples"),
        ("radial_orders", (6, 1, 6, 6, 6, 6), "radial order"),
        ("radial_orders", (6, 65, 6, 6, 6, 6), "<= 64"),
        ("radial_orders", (6,) * 5, "one more entry"),
        ("angular_limit", "both", "angular_limit"),
        ("angular_order", 0, "angular_order"),
        ("angular_order", 121, "<= 120"),
        ("angular_order", True, "angular_order"),
        ("scaling_radius", 13e-10, "scaling_radius"),
        ("scaling_radius", 0.0, "scaling_radius"),
        ("scaling_radius", math.inf, "scaling_radius"),
        ("scaling_radius", None, "irECS"),
        ("scaling_angle", None, "irECS"),
        ("infinite_order", None, "irECS"),
        ("decay", None, "irECS"),
        ("scaling_angle", math.pi / 2, "angle"),
        ("scaling_angle", 0.0, "scaling_angle"),
        ("scaling_angle", True, "scaling_angle"),
        ("infinite_order", 0, "infinite_order"),
        ("infinite_order", 65, "infinite_order"),
        ("decay", -1.0, "decay"),
        ("decay", math.nan, "decay"),
        ("cut_energy", 0.0, "cut_energy"),
        ("cut_energy", math.inf, "cut_energy"),
        ("max_step_au", -0.05, "max_step_au"),
        ("max_step_au", True, "max_step_au"),
        ("source", " ", "source"),
        ("preset", "hacc-helium-1-v1-28f0", "preset"),
        ("preset", "hacc-helium-2-v1-28f0eae146b1d665", "preset"),
        ("preset", "hacc-helium-5-v1-28f0eae146b1d665", "ions"),
        ("preset", "hacc-helium-5-fast-v1-e741d3ce524adc7e", "ions"),
        ("preset", "hacc-helium-1-slow-v1-e38292e1b30729c1", "preset"),
        ("preset", "hacc-helium-1-v1-fast-e38292e1b30729c1", "preset"),
        ("preset", 7, "preset"),
    ],
)
def test_hacc_basis_refuses_each_invalid_field(name, value, message):
    # Revalidate in place, as persisted data would be: the test session's annotation
    # checker would otherwise refuse a mistyped value before the record's own rules.
    record = _hacc()
    object.__setattr__(record, name, value)
    with pytest.raises(MaterialError, match=message):
        record.__post_init__()


def test_hacc_basis_axis_and_gauge_rules():
    basis = _hacc()
    a0 = basis.radial_edges[2]
    # R0 must be a nonzero edge, Rg an edge strictly inside R0.
    with pytest.raises(MaterialError, match="gauge_radius"):
        dataclasses.replace(basis, gauge_radius=basis.scaling_radius)
    with pytest.raises(MaterialError, match="gauge_radius"):
        dataclasses.replace(basis, gauge_radius=3 * a0)
    with pytest.raises(MaterialError, match="first radial edge"):
        dataclasses.replace(basis, radial_edges=(1e-12, *basis.radial_edges[1:]))
    with pytest.raises(MaterialError, match="strictly increasing"):
        dataclasses.replace(
            basis, radial_edges=(0.0, 2 * a0, a0, *basis.radial_edges[3:])
        )
    with pytest.raises(MaterialError, match="gauge_radius"):
        dataclasses.replace(basis, gauge_radius=math.nan)
    # The velocity-gauge control is a custom record only.
    control = dataclasses.replace(basis, gauge_radius=None)
    assert control.gauge_radius is None and not control.qualified
    with pytest.raises(MaterialError, match="gauge radius"):
        dataclasses.replace(control, preset="hacc-helium-1-v1-28f0eae146b1d665")
    # A total-L truncation is representable for helium.
    assert dataclasses.replace(basis, angular_limit="total").angular_limit == "total"


def test_hacc_basis_normalises_numbers_and_stays_immutable():
    basis = _hacc()
    integral = dataclasses.replace(
        basis, radial_edges=(0, *basis.radial_edges[1:]), cut_energy=1
    )
    assert all(type(edge) is float for edge in integral.radial_edges)
    assert type(integral.cut_energy) is float
    with pytest.raises(dataclasses.FrozenInstanceError):
        basis.ions = 2  # pyright: ignore[reportAttributeAccessIssue] - frozen check


def test_hacc_basis_labels_and_qualification_claims():
    custom = _hacc()
    assert (custom.label, custom.qualified) == ("haCC(1) (custom)", False)
    five = dataclasses.replace(custom, ions=2)
    assert five.label == "haCC(5) (custom)"
    assert (preset_basis(1).label, preset_basis(1).qualified) == ("haCC(1)", True)
    assert (preset_basis(2).label, preset_basis(2).qualified) == ("haCC(5)", True)
    fast = preset_basis(1, "fast"), preset_basis(2, "fast")
    assert [(basis.label, basis.qualified) for basis in fast] == [
        ("haCC(1) fast", True),
        ("haCC(5) fast", True),
    ]


def test_hacc_basis_variant_is_read_from_the_preset_identity():
    assert _hacc().variant is None
    assert preset_basis(1).variant == preset_basis(2).variant == "standard"
    assert preset_basis(1, "fast").variant == preset_basis(2, "fast").variant == "fast"
    # The variant is not a field: a fast record differs from the standard one in its
    # cut, step and identity only, and the standard identity is unchanged (below).
    fields = dataclasses.asdict(preset_basis(1, "fast"))
    standard = dataclasses.asdict(preset_basis(1))
    assert {name for name in fields if fields[name] != standard[name]} == {
        "cut_energy",
        "max_step_au",
        "preset",
    }


# The identities below are the SHA-256 of the canonical record; they change only
# if a field of the record or its canonical form changes (item 2).
_CUSTOM_IDENTITY = "a1aa95166f634a14e5fa342809ccbcdb2ca0ddb48dc4d8fb0c3e9e470cd84aa1"
_PRESET_IDENTITY = "0a029b10ae1b05194f7f7da3dc2d16c9bd449392c1add370c1c5e94e89159d0e"


def test_hacc_identity_is_a_stable_literal_for_preset_and_custom_records():
    assert _hacc().identity == _CUSTOM_IDENTITY
    assert preset_basis(1).identity == _PRESET_IDENTITY


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ions", 2),
        ("neutral_square", 2),
        ("radial_orders", (6, 6, 6, 6, 6, 7)),
        ("angular_limit", "total"),
        ("angular_order", 3),
        ("scaling_angle", 0.2),
        ("infinite_order", 7),
        ("decay", 2.0e10),
        ("gauge_radius", None),
        ("cut_energy", 1e-17),
        ("max_step_au", 0.04),
        ("source", "another reason"),
    ],
)
def test_hacc_identity_covers_every_field(name, value):
    basis = _hacc()
    assert dataclasses.replace(basis, **{name: value}).identity != basis.identity


def test_hacc_identities_separate_models_presets_and_sae():
    one, five = preset_basis(1), preset_basis(2)
    assert one.identity != five.identity
    fast = {preset_basis(ions, "fast").identity for ions in (1, 2)}
    assert len(fast) == 2 and not fast & {one.identity, five.identity}
    # A custom record equal to a preset in every number differs only by `preset`.
    copy = dataclasses.replace(one, preset=None)
    assert copy.identity != one.identity
    assert dataclasses.asdict(copy) | {"preset": one.preset} == dataclasses.asdict(one)
    # No haCC identity equals the SAE identity of a record with the same numbers.
    sae = TDSEBasis(
        "same numbers",
        one.radial_edges,
        one.radial_orders,
        one.angular_order,
        one.cut_energy,
        source=one.source,
        scaling_radius=one.scaling_radius,
        scaling_angle=one.scaling_angle,
        infinite_order=one.infinite_order,
        decay=one.decay,
    )
    assert sae.identity != one.identity
    assert sae.identity != copy.identity
    assert one.radial_edges[-1] == 80 * A0


def _hacc_response(**changes):
    return hacc_response(**changes)


def test_hacc_response_resolves_its_defaults_per_model(response):
    hacc = _hacc_response(atomic_window="auto")
    assert hacc.atomic_window == TDSEAtomicWindow()
    assert hacc.consistency == tdse.TDSEConsistency()
    sae = dataclasses.replace(response, kerr_model="tdse_native")
    assert sae.atomic_window is None
    # Omission and "auto" store the same resolved record; explicit None is kept.
    assert _hacc_response(atomic_window=None).atomic_window is None
    windowed = dataclasses.replace(sae, atomic_window=TDSEAtomicWindow(60e-15))
    assert windowed.atomic_window == TDSEAtomicWindow(60e-15)


@pytest.mark.parametrize("scale", [1.537, 2.0, math.nextafter(1.0, 2.0)])
def test_hacc_response_refuses_any_remainder_scale_but_one(scale):
    with pytest.raises(MaterialError, match="remainder_scale"):
        _hacc_response(remainder_scale=scale)


@pytest.mark.parametrize("model", ["optmat_cubic", "tdse_shadow"])
def test_hacc_response_refuses_cubic_policies(model):
    basis = tiny_basis()
    with pytest.raises(MaterialError, match="third order"):
        TDSEGasResponse(
            "helium",
            basis,
            basis.max_step_au * tdse.AU_TIME,
            model,
            1.0,
            hacc_cases.calibration(basis),
            TDSESampling(),
            cubic_scale=1.5 if model == "tdse_shadow" else None,
            model="hacc",
        )


def test_hacc_response_rules_of_the_section_5_1_table(response, basis):
    hacc_basis = tiny_basis()
    step = hacc_basis.max_step_au * tdse.AU_TIME
    common = (
        step,
        "tdse_native",
        1.0,
        hacc_cases.calibration(hacc_basis),
        TDSESampling(),
    )
    with pytest.raises(MaterialError, match="requires a HaccBasis"):
        TDSEGasResponse("helium", basis, *common, model="hacc")
    with pytest.raises(MaterialError, match="requires a TDSEBasis"):
        TDSEGasResponse("helium", hacc_basis, *common)
    with pytest.raises(MaterialError, match="cubic_scale"):
        _hacc_response(cubic_scale=1.0)
    with pytest.raises(MaterialError, match="max_atomic_step"):
        _hacc_response(max_atomic_step=step * (1 + 1e-12))
    # Two units in the last place of the conversion are admitted.
    nearby = math.nextafter(math.nextafter(step, 1.0), 1.0)
    assert _hacc_response(max_atomic_step=nearby).max_atomic_step == nearby
    with pytest.raises(MaterialError, match="haCC responses only"):
        dataclasses.replace(response, consistency=tdse.TDSEConsistency(warn=0.05))
    with pytest.raises(MaterialError, match="basis_identity"):
        _hacc_response(calibration=hacc_cases.calibration(tiny_basis(ions=2)))
    # Mistyped persisted values are refused by the record's own revalidation.
    for name, value, message in (
        ("model", "tdcis", "model must be"),
        ("atomic_window", "full", "atomic_window"),
        ("atomic_window", 60e-15, "atomic_window"),
        ("consistency", (0.1, 0.3), "TDSEConsistency"),
        ("basis", "haCC(1)", "basis and calibration"),
    ):
        record = _hacc_response()
        object.__setattr__(record, name, value)
        with pytest.raises(MaterialError, match=message):
            record.__post_init__()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"duration": 0.0}, "duration"),
        ({"duration": -1e-15}, "duration"),
        ({"duration": math.inf}, "duration"),
        ({"duration": True}, "duration"),
        ({"taper": 0.0}, "taper"),
        ({"taper": math.nan}, "taper"),
        ({"taper": True}, "taper"),
        ({"duration": 10e-15, "taper": 5e-15}, "half"),
        ({"duration": 10e-15, "taper": 6e-15}, "half"),
    ],
)
def test_atomic_window_validation(values, message):
    record = TDSEAtomicWindow(duration=40e-15)
    for name, value in values.items():
        object.__setattr__(record, name, value)
    with pytest.raises(MaterialError, match=message):
        record.__post_init__()


def test_atomic_window_defaults_and_normalisation():
    window = TDSEAtomicWindow()
    assert (window.duration, window.taper) == (None, 5e-15)
    assert TDSEAtomicWindow(duration=60e-15).duration == 60e-15
    assert type(TDSEAtomicWindow(duration=1, taper=0.25).duration) is float


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"warn": 0.0}, "warn"),
        ({"warn": -0.1}, "warn"),
        ({"warn": math.inf}, "warn"),
        ({"warn": True}, "warn"),
        ({"refuse": math.nan}, "refuse"),
        ({"refuse": False}, "refuse"),
        ({"warn": 0.4, "refuse": 0.3}, "exceed"),
        ({"enforce": 0}, "enforce"),
        ({"enforce": "no"}, "enforce"),
    ],
)
def test_consistency_policy_validation(values, message):
    record = tdse.TDSEConsistency()
    for name, value in values.items():
        object.__setattr__(record, name, value)
    with pytest.raises(MaterialError, match=message):
        record.__post_init__()


def test_consistency_policy_defaults_and_equal_bounds():
    policy = tdse.TDSEConsistency()
    assert (policy.warn, policy.refuse, policy.enforce) == (0.1, 0.3, True)
    assert tdse.TDSEConsistency(warn=0.2, refuse=0.2).refuse == 0.2
    assert _hacc_response(consistency=tdse.TDSEConsistency(enforce=False))

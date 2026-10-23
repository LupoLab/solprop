# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the public tensor response against the independent oracle sums.

The molecular responses are the supplied route's, built from solprop's former Raman
constants (``former_raman``, from its registry at dc4ca0d): six kernels of different
shape (rotational ladders with and without Dicke narrowing, spherical tops) on which
these bounds were set. optmat's built-in gas responses are checked against the same
kinds of sums in test_optmat_raman.py.
"""

from dataclasses import FrozenInstanceError, fields, replace
from inspect import unwrap

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bounds import assert_within, fft_bound, gamma
from former_raman import (
    FORMER_CH4,
    FORMER_D2,
    FORMER_H2,
    FORMER_N2,
    FORMER_O2,
    FORMER_SF6,
    former_response,
)
from solprop import fourier
from solprop.backends.numpy_ref.passes import raman_polarisation
from solprop.constants import EPS0
from solprop.exceptions import GridError, MaterialError
from solprop.nonlinear.raman import (
    DampedOscillator,
    MolecularOptions,
    RamanResponse,
    RamanTensorTerm,
    _check_delays,
    _check_evaluate,
    response_for,
    undamped_dephasing,
)
from solprop.output.schema import serialise
from solprop.physdata import MolecularRaman
from vector_raman_declaration import Declaration
from vector_raman_tensor_oracle import (
    JONES,
    cartesian_direct,
    causal_sum,
    circular_direct,
    convolution_bound,
)

MOLECULES = [
    pytest.param(constants, id=str(getattr(constants.gas, "value", constants.gas)))
    for constants in (
        FORMER_H2,
        FORMER_D2,
        FORMER_N2,
        FORMER_O2,
        FORMER_CH4,
        FORMER_SF6,
    )
]
"""solprop's former molecular constants (dc4ca0d), supplied as a caller's data."""


@pytest.mark.parametrize("material", MOLECULES)
@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("count", [17, 97])
def test_material_tensor_matches_independent_causal_sum(
    material: MolecularRaman, envelope, count
):
    rng = np.random.default_rng(4002)
    field = 2e8 * rng.standard_normal((2, count))
    if envelope:
        field = field + 1j * 2e8 * rng.standard_normal(field.shape)
    dt, density = 1e-15, 1e25
    response = former_response(material)
    vibration, rotation = response.tensor_impulse(np.arange(count) * dt, density)
    expected = (
        EPS0
        * density
        * cartesian_direct(field, vibration, rotation, dt, envelope=envelope)
    )
    actual = response.evaluate(
        field, density, kind="envelope" if envelope else "field", dt=dt
    )
    bound = (
        EPS0
        * density
        * convolution_bound(field, vibration, rotation, dt, envelope=envelope, fft=True)
    )
    assert bound > 0
    assert_allclose(actual, expected, atol=bound, rtol=0)
    angle = 0.713
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = response.evaluate(
        rotate @ field, density, kind="envelope" if envelope else "field", dt=dt
    )
    # Two evaluations and the real rotation contribute at most four base budgets.
    assert_allclose(rotated, rotate @ actual, atol=4 * bound, rtol=0)
    if envelope:
        circular = JONES.conj().T @ field
        expected_circular = (
            EPS0 * density * circular_direct(circular, vibration, rotation, dt)
        )
        assert_allclose(
            JONES.conj().T @ actual, expected_circular, atol=4 * bound, rtol=0
        )


@pytest.mark.parametrize("material", MOLECULES)
@pytest.mark.parametrize("envelope", [False, True])
def test_tensor_scalar_reduction_preserves_legacy_arithmetic(
    material: MolecularRaman, envelope
):
    response = former_response(material)
    legacy = RamanResponse(response.models)
    t = np.arange(97) * 1e-15
    scalar = 2e8 * np.sin(np.pi * t / t[-1]) ** 2
    if envelope:
        scalar = scalar * np.exp(0.6j * t / t[-1])
    kind = "envelope" if envelope else "field"
    old = legacy.evaluate(scalar[None], 1e25, kind=kind, dt=1e-15)
    same = response.evaluate(scalar[None], 1e25, kind=kind, dt=1e-15)
    # One convolution by two routes: a padded round trip and its products.
    assert_within(same, old, fft_bound(2 * t.size))
    direction = np.array([np.cos(0.37), np.sin(0.37)])[:, None]
    vector = direction * scalar
    actual = response.evaluate(vector, 1e25, kind=kind, dt=1e-15)
    hv, hr = response.tensor_impulse(t, 1e25)
    bound = (
        EPS0
        * 1e25
        * convolution_bound(vector, hv, hr, 1e-15, envelope=envelope, fft=True)
    )
    assert_allclose(actual, direction * old, atol=bound, rtol=0)


def test_beat_removed_tensor_uses_analytic_drive_and_real_final_field():
    dt, density = 1e-15, 1e25
    time = np.arange(97) * dt
    field = 2e8 * np.array(
        [
            np.exp(-(((time - 35e-15) / 12e-15) ** 2)) * np.cos(1.4e15 * time),
            np.exp(-(((time - 45e-15) / 15e-15) ** 2)) * np.sin(1.4e15 * time),
        ]
    )
    response = former_response(FORMER_H2, thg=False)
    analytic = np.array([fourier.analytic_signal(row) for row in field])
    hv, hr = response.tensor_impulse(time, density)
    q = 0.5 * np.real(analytic[:, None] * analytic[None].conj())
    trace = q[0, 0] + q[1, 1]
    expected = np.zeros_like(field)
    for i in range(2):
        expected[i] += causal_sum(hv, trace, dt) * field[i]
        for j in range(2):
            expected[i] += (
                0.5
                * causal_sum(hr, 3 * q[i, j] - (trace if i == j else 0), dt)
                * field[j]
            )
    expected *= EPS0 * density
    bound = (
        EPS0
        * density
        * convolution_bound(analytic, hv, hr, dt, envelope=True, fft=True)
    )
    assert_allclose(
        response.evaluate(field, density, kind="field", dt=dt),
        expected,
        atol=bound,
        rtol=0,
    )


@pytest.mark.parametrize("helicity", [-1, 1])
@pytest.mark.parametrize("channel, factor", [("isotropic", 1.0), ("rank2", 0.25)])
def test_fixed_circular_limit_for_explicit_custom_terms(helicity, channel, factor):
    time = np.arange(97) * 0.07
    model = DampedOscillator(1.0, 1.3, undamped_dephasing())
    response = RamanResponse(tensor_terms=(RamanTensorTerm(model, channel),))
    pulse = np.exp(-(((time - 3) / 1.2) ** 2) + 0.4j * time)
    field = np.array([pulse, helicity * 1j * pulse]) / np.sqrt(2)
    kernel = model.undamped(time)
    expected = EPS0 * factor * causal_sum(kernel, abs(pulse) ** 2 / 2, 0.07) * field
    hv, hr = response.tensor_impulse(time, 1.0)
    bound = EPS0 * convolution_bound(field, hv, hr, 0.07, envelope=True, fft=True)
    assert_allclose(
        response.evaluate(field, 1.0, kind="envelope", dt=0.07),
        expected,
        atol=bound,
        rtol=0,
    )


def test_canonical_terms_replacement_reconstruction_and_legacy_records():
    response = former_response(FORMER_H2)
    assert {field.name for field in fields(response)} == {
        "tensor_terms",
        "fraction",
        "thg",
        "species",
    }
    assert replace(response, species="hydrogen").tensor_terms == response.tensor_terms
    copied = RamanResponse(
        **{field.name: getattr(response, field.name) for field in fields(response)}
    )
    assert copied == response
    legacy = RamanResponse(models=response.models)
    assert serialise(legacy) == Declaration(response.models).record()
    record = serialise(response)
    assert isinstance(record, dict)
    assert "models" not in record and "tensor_terms" in record
    assert serialise(record) == record
    changed = replace(
        response,
        tensor_terms=(
            replace(response.tensor_terms[0], contraction="isotropic"),
            *response.tensor_terms[1:],
        ),
    )
    assert serialise(changed) != record
    with pytest.raises(FrozenInstanceError):
        response.__setattr__("species", "changed")
    assert response.traits.polarisation == "tensor"
    assert legacy.traits.polarisation == "scalar"


@pytest.mark.parametrize(
    "arguments",
    [{}, {"models": ()}, {"tensor_terms": ()}, {"models": (), "tensor_terms": ()}],
)
def test_empty_or_ambiguous_constructor_routes_fail(arguments):
    with pytest.raises(MaterialError):
        RamanResponse(**arguments)


@pytest.mark.parametrize("count", [0, 3])
def test_invalid_physical_component_count_fails(count):
    with pytest.raises(GridError, match="between one and 2 physical components"):
        _check_evaluate(np.zeros((count, 16)), "field", 1e-15)


@pytest.mark.parametrize("density", [-1.0, np.inf, np.nan])
def test_nonfinite_or_negative_density_fails_before_evaluation(density):
    response = former_response(FORMER_H2)
    for evaluate in (
        lambda: response.impulse(np.arange(16, dtype=float), density),
        lambda: response.tensor_impulse(np.arange(16, dtype=float), density),
        lambda: response.evaluate(np.zeros((2, 16)), density, kind="field", dt=1e-15),
    ):
        with pytest.raises(MaterialError, match="finite, non-negative"):
            evaluate()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_field_fails_even_at_vacuum(bad):
    field = np.zeros((2, 16))
    field[1, 0] = bad
    with pytest.raises(GridError, match="finite numbers"):
        former_response(FORMER_H2).evaluate(field, 0.0, kind="field", dt=1e-15)


def test_nonnumeric_fields_and_invalid_delays_fail():
    with pytest.raises(GridError, match="finite numbers"):
        _check_evaluate(np.full((2, 16), "bad"), "field", 1e-15)
    # Public beartype annotations reject invalid ranks/dtypes first. Exercise the
    # explicit boundary validators with equivalent numeric nonfinite samples.
    with pytest.raises(GridError, match="finite and real"):
        _check_delays(np.array([np.nan]))


@pytest.mark.parametrize(
    "delays", [np.zeros((2, 2)), np.array(["bad"]), np.array([1j])]
)
def test_delay_validation_without_optional_runtime_type_hook(delays):
    # Normal Python users do not install the test-only annotation hook. Reach
    # the same explicit boundary checks without that hook rejecting first.
    with pytest.raises(GridError, match="Raman delays"):
        unwrap(_check_delays)(delays)


@pytest.mark.parametrize(
    "attribute, value, message",
    [("model", object(), "RamanModel"), ("contraction", "unknown", "unknown Raman")],
)
def test_term_validation_rejects_invalid_protocol_or_role(attribute, value, message):
    term = RamanTensorTerm(former_response(FORMER_H2).models[0], "rank2")
    object.__setattr__(term, attribute, value)
    with pytest.raises(MaterialError, match=message):
        term.__post_init__()


def test_constructor_rejects_unlabelled_items_without_runtime_type_hook():
    response = object.__new__(RamanResponse)
    with pytest.raises(MaterialError, match="RamanTensorTerm declarations"):
        unwrap(RamanResponse.__init__)(response, tensor_terms=(object(),))


@pytest.mark.parametrize("mixed", [False, True])
def test_scalar_only_term_refuses_vector_and_tensor_impulse(mixed):
    model = response_for("SiO2").models[0]
    terms = (RamanTensorTerm(model, "scalar_only"),)
    if mixed:
        terms += former_response(FORMER_H2).tensor_terms
    response = RamanResponse(tensor_terms=terms)
    with pytest.raises(MaterialError, match="scalar_only"):
        response.tensor_impulse(np.arange(16, dtype=float), 0.0)
    with pytest.raises(GridError, match="scalar_only"):
        response.evaluate(np.zeros((2, 16)), 0.0, kind="field", dt=1e-15)


def test_factory_selection_carries_explicit_roles():
    assert [term.contraction for term in former_response(FORMER_H2).tensor_terms] == [
        "rank2",
        "isotropic",
    ]
    assert [
        term.contraction
        for term in former_response(
            FORMER_H2, MolecularOptions(vibration=False)
        ).tensor_terms
    ] == ["rank2"]
    assert [term.contraction for term in former_response(FORMER_SF6).tensor_terms] == [
        "isotropic"
    ]
    assert not response_for("SiO2").tensor_capable


def test_zero_field_and_vacuum_give_exact_zero():
    response = former_response(FORMER_H2)
    for field, density in ((np.zeros((2, 17)), 1e25), (np.ones((2, 17)), 0.0)):
        assert np.all(response.evaluate(field, density, kind="field", dt=1e-15) == 0)


@pytest.mark.parametrize("rank2", [False, True])
def test_delta_and_zero_kernel_limits_reduce_to_local_tensor(rank2):
    rng = np.random.default_rng(4051)
    count, dt = 17, 0.07
    field = rng.standard_normal((2, count)) + 1j * rng.standard_normal((2, count))
    hv, hr = np.zeros(count), np.zeros(count)
    hv[0], hr[0] = 0.7 / dt, (1.1 if rank2 else 0.0) / dt
    spectra = np.fft.rfft(np.array([hv + hr / 4, 3 * hr / 4]), n=2 * count) * dt
    q = np.real(field[:, None] * field[None].conj()) / 2
    trace = q[0, 0] + q[1, 1]
    matrix = 0.7 * trace * np.eye(2)[:, :, None]
    if rank2:
        matrix += 0.55 * (3 * q - trace * np.eye(2)[:, :, None])
    expected = EPS0 * np.einsum("ijt,jt->it", matrix, field)
    bound = EPS0 * convolution_bound(field, hv, hr, dt, envelope=True, fft=True)
    assert bound > 0
    actual = raman_polarisation(
        field[None], spectra, kind="envelope", thg=False, density=1.0
    )[0]
    assert_allclose(actual, expected, rtol=0, atol=bound)
    zero = raman_polarisation(
        field[None], np.zeros_like(spectra), kind="envelope", thg=False, density=1.0
    )
    np.testing.assert_array_equal(zero, np.zeros_like(zero))


def test_reduced_positive_frequency_system_has_zero_photon_number_derivative():
    rng = np.random.default_rng(4052)
    count, dt = 97, 0.07
    field = rng.standard_normal((2, count)) + 1j * rng.standard_normal((2, count))
    response = RamanResponse(
        tensor_terms=(
            RamanTensorTerm(
                DampedOscillator(0.7, 2.3, undamped_dephasing()), "isotropic"
            ),
            RamanTensorTerm(DampedOscillator(1.1, 0.9, undamped_dephasing()), "rank2"),
        )
    )
    source = response.evaluate(field, 1.0, kind="envelope", dt=dt)
    frequency = 100 + 2 * np.pi * np.fft.fftfreq(count, dt)
    assert np.all(frequency > 0)
    spectrum, polarisation = np.fft.fft(field), np.fft.fft(source)
    derivative = -1j * frequency * polarisation
    summands = spectrum.conj() * derivative / frequency
    # For dA/dz=-i*kappa*omega*FFT(M A), real symmetric M and real constant
    # kappa, Parseval gives d(sum|A|^2/omega)/dz=0. Two FFT errors plus
    # complex products and the 2*N-term reduction bound the computed residual.
    relative = 128 * np.finfo(float).eps * np.log2(2 * count) + 2 * gamma(8 * count)
    bound = 2 * relative * np.sum(abs(summands))
    assert bound > 0
    assert abs(2 * np.real(np.sum(summands))) <= bound

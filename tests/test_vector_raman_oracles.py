# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the independent vector Raman derivation before changing the production seam.

The molecular kernels are the supplied route's, from solprop's former Raman constants
(``former_raman``, from its registry at dc4ca0d), on which these budgets were set.
"""

import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

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
from solprop.constants import EPS0
from solprop.nonlinear.raman import MolecularOptions, RamanResponse
from solprop.output.schema import serialise
from solprop.physdata import MolecularRaman
from vector_raman_declaration import Declaration, Term
from vector_raman_tensor_oracle import (
    BUDGETS,
    EPS,
    JONES,
    algebra_experiments,
    cartesian_direct,
    causal_sum,
    circular_direct,
    convolution_bound,
    gamma,
    orientation_tensor,
    oscillator_experiment,
    three_drive_fft,
)


def test_proposed_declaration_preserves_one_authority_and_scalar_metadata():
    existing = RamanResponse(models=former_response(FORMER_H2).models)
    legacy = Declaration(
        existing.models, existing.fraction, existing.thg, existing.species
    )
    assert legacy.record() == serialise(existing)
    tensor = Declaration(
        tensor_terms=(
            Term(existing.models[0], "rank2"),
            Term(existing.models[1], "isotropic"),
        )
    )
    copied = Declaration(
        tensor_terms=tensor.tensor_terms,
        fraction=tensor.fraction,
        thg=tensor.thg,
        species=tensor.species,
    )
    assert copied == tensor
    assert copied.models == existing.models
    assert replace(tensor, species="H2").tensor_terms == tensor.tensor_terms
    changed = replace(
        tensor,
        tensor_terms=(
            replace(tensor.tensor_terms[0], contraction="isotropic"),
            *tensor.tensor_terms[1:],
        ),
    )
    assert json.dumps(changed.record(), sort_keys=True) != json.dumps(
        tensor.record(), sort_keys=True
    )
    assert "models" not in tensor.record()
    assert serialise(tensor.record()) == tensor.record()
    with pytest.raises(FrozenInstanceError):
        tensor.__setattr__("species", "H2")


@pytest.mark.parametrize(
    "arguments",
    [{}, {"models": ()}, {"tensor_terms": ()}, {"models": (), "tensor_terms": ()}],
)
def test_proposed_declaration_refuses_ambiguous_or_empty_routes(arguments):
    with pytest.raises(ValueError):
        Declaration(**arguments)


def test_orientation_quadrature_matches_traceless_three_dimensional_tensor():
    actual = orientation_tensor()
    identity = np.eye(3)
    expected = 0.75 * (
        np.einsum("ik,jl->ijkl", identity, identity)
        + np.einsum("il,jk->ijkl", identity, identity)
    ) - 0.5 * np.einsum("ij,kl->ijkl", identity, identity)
    # 96 angles, up to 64 arithmetic operations per fourth-order moment.
    bound = gamma(64 * 96)
    assert_allclose(actual, expected, rtol=0, atol=bound)
    assert_allclose(np.einsum("iikl->kl", actual), 0, rtol=0, atol=3 * bound)


def test_complex_histories_obey_orientation_rotation_and_circular_identities():
    result = algebra_experiments()
    assert result["direct_bound"] > 0
    for name in ("orientation_error", "circular_error", "rotation_error"):
        assert result[name] < result["direct_bound"]
    assert 0 < result["fft_error"] < result["fft_bound"]


@pytest.mark.parametrize(
    "material",
    [
        pytest.param(constants, id=str(getattr(constants.gas, "value", constants.gas)))
        for constants in (
            FORMER_H2,
            FORMER_D2,
            FORMER_N2,
            FORMER_O2,
            FORMER_CH4,
            FORMER_SF6,
        )
    ],
)
@pytest.mark.parametrize("angle", [0.0, np.pi / 2, 0.37])
@pytest.mark.parametrize("envelope", [False, True])
def test_tensor_linear_reduction_preserves_each_existing_material(
    material: MolecularRaman, angle, envelope
):
    time = np.arange(97) * 1e-15
    scalar = 2e8 * np.sin(np.pi * time / time[-1]) ** 2
    if envelope:
        scalar = scalar * np.exp(0.4j * time / time[-1])
    field = np.array([np.cos(angle), np.sin(angle)])[:, None] * scalar
    density = 1e25
    vibration = former_response(material, MolecularOptions(rotation=False)).impulse(
        time, density
    )
    rotation = np.zeros_like(time)
    if material.rotational is not None:
        rotation = former_response(material, MolecularOptions(vibration=False)).impulse(
            time, density
        )
    actual = (
        EPS0
        * density
        * cartesian_direct(field, vibration, rotation, time[1], envelope=envelope)
    )
    scalar_result = former_response(material).evaluate(
        scalar[None, :], density, kind="envelope" if envelope else "field", dt=time[1]
    )[0]
    reference = np.array([np.cos(angle), np.sin(angle)])[:, None] * scalar_result
    # Absolute summands bound both contractions and the production scalar FFT.
    bound = (
        EPS0
        * density
        * convolution_bound(
            field, vibration, rotation, time[1], envelope=envelope, fft=True
        )
    )
    assert np.max(np.abs(reference)) > 0
    assert_allclose(actual, reference, rtol=0, atol=bound)


@pytest.mark.parametrize("handedness", [-1, 1])
def test_fixed_circular_rotation_is_one_quarter_at_equal_total_power(handedness):
    time = np.arange(97) * 0.07
    pulse = np.exp(-(((time - 3) / 1.2) ** 2) + 0.4j * time)
    field = np.array([pulse, handedness * 1j * pulse]) / np.sqrt(2)
    kernel = np.sin(time) * np.exp(-0.1 * time)
    zero = np.zeros_like(time)
    for vibration, rotation, factor in ((kernel, zero, 1), (zero, kernel, 0.25)):
        actual = cartesian_direct(field, vibration, rotation, 0.07, envelope=True)
        reference = factor * causal_sum(kernel, abs(pulse) ** 2 / 2, 0.07) * field
        bound = convolution_bound(field, vibration, rotation, 0.07, envelope=True)
        assert_allclose(actual, reference, rtol=0, atol=bound)
    # Unitary circular coordinates: no population in the opposite helicity.
    assert_allclose(
        (JONES.conj().T @ field)[(1 + handedness) // 2], 0, rtol=0, atol=8 * EPS
    )


def test_circular_equivalence_requires_delayed_off_diagonal_coherence():
    time = np.arange(97) * 0.07
    amplitudes = np.array(
        [
            np.exp(-(((time - 2) / 1.3) ** 2) + 0.6j * time),
            np.exp(-(((time - 3) / 1.3) ** 2) - 0.3j * time),
        ]
    )
    rotation = np.sin(time) * np.exp(-0.03 * time)
    actual = circular_direct(amplitudes, np.zeros_like(time), rotation, 0.07)
    omitted = (
        causal_sum(rotation / 8, np.sum(abs(amplitudes) ** 2, axis=0), 0.07)
        * amplitudes
    )
    # Negative control: dropping coherence is an order-one physics error.
    assert np.linalg.norm(actual - omitted) > 0.1 * np.linalg.norm(actual)


def test_delayed_orthogonal_probe_retains_signed_rotational_channel():
    time = np.arange(129) * 0.07
    pump = np.where((time > 1) & (time < 3), np.sin(np.pi * (time - 1) / 2) ** 2, 0)
    probe = 0.01 * np.where(
        (time > 4) & (time < 6), np.sin(np.pi * (time - 4) / 2) ** 2, 0
    )
    field = np.array([pump, probe], dtype=complex)
    rotation = np.sin(0.7 * time) * np.exp(-0.03 * time)
    zero = np.zeros_like(time)
    actual = cartesian_direct(field, zero, rotation, 0.07, envelope=True)
    alone = cartesian_direct(
        np.array([zero, probe], dtype=complex), zero, rotation, 0.07, envelope=True
    )
    expected = -0.25 * causal_sum(rotation, pump**2, 0.07) * probe
    bound = convolution_bound(field, zero, rotation, 0.07, envelope=True)
    assert_allclose(actual[1] - alone[1], expected, rtol=0, atol=bound)
    assert np.max(abs(expected)) > 0
    assert np.all(actual[:, time < 1] == 0)


@pytest.mark.parametrize("count", [17, 64, 97])
@pytest.mark.parametrize("envelope", [False, True])
def test_tensor_convolution_padding_tail_and_window_extension(count, envelope):
    time = np.arange(count) * 0.07
    rng = np.random.default_rng(4002)
    field = rng.normal(size=(2, count))
    if envelope:
        field = field + 1j * rng.normal(size=(2, count))
    vibration, rotation = 0.7 * np.sin(2.3 * time), -np.sin(1.2 * time)
    direct = cartesian_direct(field, vibration, rotation, 0.07, envelope=envelope)
    fft = three_drive_fft(field, vibration, rotation, 0.07, envelope=envelope)
    bound = convolution_bound(
        field, vibration, rotation, 0.07, envelope=envelope, fft=True
    )
    assert_allclose(fft, direct, rtol=0, atol=bound)
    extended_time = np.arange(2 * count) * 0.07
    extended = three_drive_fft(
        np.pad(field, ((0, 0), (0, count))),
        0.7 * np.sin(2.3 * extended_time),
        -np.sin(1.2 * extended_time),
        0.07,
        envelope=envelope,
    )
    assert_allclose(extended[:, :count], direct, rtol=0, atol=2 * bound)
    # Inspect the convolution itself: multiplying by a zero early field could
    # otherwise hide wraparound from a last-sample excitation.
    drive = np.zeros(count)
    drive[-1] = 1
    transformed = np.fft.irfft(
        np.fft.rfft(drive, n=2 * count) * np.fft.rfft(rotation, n=2 * count),
        n=2 * count,
    )
    assert_allclose(
        transformed[:count],
        0,
        rtol=0,
        atol=64 * EPS * np.ceil(np.log2(2 * count)) * np.sum(abs(rotation)),
    )


def test_beat_removed_field_drive_matches_explicit_optical_cycle_average():
    count = 128
    time = 2 * np.pi * np.arange(count) / count
    envelope = np.array([1.2 * np.exp(2j * time), 0.7 * np.exp(1j * (3 * time + 0.37))])
    analytic = fourier.analytic_signal(np.real(envelope * np.exp(17j * time)))
    actual = 0.5 * np.real(analytic[:, None] * analytic[None, :].conj())
    phases = 2 * np.pi * np.arange(16) / 16
    sampled = (envelope[None, :, :] * np.exp(1j * phases[:, None, None])).real
    expected = np.einsum("pit,pjt->ijt", sampled, sampled) / 16
    # Hilbert transform plus cycle reduction at a band-limited periodic carrier.
    assert_allclose(actual, expected, rtol=0, atol=gamma(64 * count) * 1.2**2)


def test_independent_tensor_zero_field_and_vacuum_remain_finite():
    time = np.arange(17) * 0.07
    kernel = np.sin(time)
    zero = cartesian_direct(
        np.zeros((2, 17), dtype=complex), kernel, kernel, 0.07, envelope=True
    )
    vacuum = 0 * three_drive_fft(
        np.ones((2, 17), dtype=complex), kernel, kernel, 0.07, envelope=True
    )
    assert np.all(zero == 0)
    assert np.all(vacuum == 0)


@pytest.mark.slow
def test_tensor_direct_sums_converge_to_independently_driven_oscillators():
    result = oscillator_experiment()
    assert result["ode_repeat_error"] < BUDGETS["oscillator_repeat_relative_bound"]
    assert result["order_pass"], result

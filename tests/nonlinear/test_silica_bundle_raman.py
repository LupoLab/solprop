# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Silica's Raman response from optmat's bundle on the existing seam (SEP 3, 4.5).

The scalar term is the bundle's copolarised shape sampled into one undamped row with
no dephasing; the tensor terms are Hellwarth's ``a + b/3`` and ``2 b/3`` with the
seam's weights ``(1, 0)`` and ``(1/4, 3/4)``, so a copolarised probe sees ``a + b``
and a cross-polarised one ``a``. The bounds are rounding statements unless a
quadrature is named.
"""

import numpy as np
import pytest

from former_raman import FORMER_SILICA_LINES, FORMER_SILICA_TENSOR
from solprop.constants import EPS0
from solprop.exceptions import MaterialError, TranslationError
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.raman import (
    IntermediateBroadening,
    SampledShape,
    SilicaTensorResponse,
    bundle_response,
)
from solprop.physdata.glasses import SilicaBundle
from solprop.profiles import Constant
from solprop.translate.responses import raman_coefficients

DEFAULT = SilicaBundle.resolve()
TENSOR = SilicaBundle.resolve(tensor=True)
STEP = 2e-16
DELAYS = np.arange(10000) * STEP


def test_the_scalar_shape_is_the_bundle_response_with_no_dephasing():
    shape = SampledShape(DEFAULT, amplitude=3.0)
    assert shape.dephasing.decay_rate(1e26) == 0.0
    positive = DELAYS[1:]
    np.testing.assert_array_equal(shape.undamped(positive), 3.0 * DEFAULT.h_r(positive))
    # A shape that opens with a sine is exactly zero at zero delay and below it.
    assert shape.undamped(np.array([-1e-15, 0.0])).tolist() == [0.0, 0.0]
    assert shape.name == "silica_bundle/default:parallel"
    assert shape.material_records == DEFAULT.material_records


def test_optmat_hollenbeck_is_solprops_hollenbeck_to_rounding():
    # The same published line set (Hollenbeck and Cantrell 2002), normalised by two
    # independent Faddeeva integrals: agreement to a few roundings. Ours is the line
    # set of solprop's former registry (dc4ca0d), supplied as a caller's data.
    ours = IntermediateBroadening(FORMER_SILICA_LINES).undamped(DELAYS)
    theirs = SampledShape(DEFAULT).undamped(DELAYS)
    np.testing.assert_allclose(
        theirs, ours, rtol=0, atol=float(2e-15 * np.max(abs(ours)))
    )


def test_a_jump_at_zero_takes_half_its_right_hand_limit():
    rank = SampledShape(TENSOR, "rank2")
    # b(0+) = 2 f_b / tau_b, Lin and Agrawal's boson peak; the trapezoid's endpoint.
    limit = 2.0 * 0.21 / 96e-15
    assert float(rank.undamped(np.array([0.0]))[0]) == pytest.approx(
        0.5 * (2 / 3) * limit, rel=1e-15
    )
    assert float(SampledShape(TENSOR, "isotropic").undamped(np.array([0.0]))[0]) == (
        pytest.approx(0.5 * limit / 3, rel=1e-15)
    )


def test_the_channels_are_hellwarths_a_plus_b_over_three_and_two_b_over_three():
    a, b = TENSOR.h_r(DELAYS[1:], "a"), TENSOR.h_r(DELAYS[1:], "b")
    isotropic = SampledShape(TENSOR, "isotropic").undamped(DELAYS[1:])
    rank = SampledShape(TENSOR, "rank2").undamped(DELAYS[1:])
    # The same two combinations of the channels: a couple of roundings each.
    np.testing.assert_allclose(isotropic, a + b / 3.0, rtol=4 * np.finfo(float).eps)
    np.testing.assert_allclose(rank, 2.0 * b / 3.0, rtol=4 * np.finfo(float).eps)
    scale = float(np.max(abs(a + b)))
    # Copolarised: the isotropic and rank-two kernels add (the trace and T(Q) of a
    # fixed linear polarisation both reduce to the field's square).
    np.testing.assert_allclose(isotropic + rank, a + b, rtol=0, atol=4e-16 * scale)
    # Cross-polarised: T(Q)_yy = -tr(Q)/2 for a pump along x, so the probe sees a.
    np.testing.assert_allclose(isotropic - rank / 2, a, rtol=0, atol=4e-16 * scale)


def test_tensor_channels_of_a_copolarised_shape_are_refused_by_name():
    for channel in ("isotropic", "rank2"):
        with pytest.raises(MaterialError, match=r"silica_bundle/default .* only"):
            SampledShape(DEFAULT, channel)
    # The runtime type hook refuses these first in the test session; production
    # has no hook, so the dataclass's own check is what refuses them there.
    for attribute, value, message in (
        ("channel", "trace", "unknown silica Raman channel"),
        ("amplitude", float("nan"), "must be finite"),
    ):
        shape = SampledShape(DEFAULT)
        object.__setattr__(shape, attribute, value)
        with pytest.raises(MaterialError, match=message):
            shape.__post_init__()
    with pytest.raises(MaterialError, match=r"vector run .* copolarised only"):
        bundle_response(DEFAULT, tensor=True)


def test_the_scalar_and_tensor_responses_on_the_seam():
    scalar = bundle_response(DEFAULT, amplitude=2.0, fraction=0.5, thg=False)
    assert not scalar.tensor_capable and scalar.fraction == 0.5
    tensor = bundle_response(
        TENSOR, amplitude=2.0, fraction=TENSOR.raman_fraction, tensor=True
    )
    assert tensor.tensor_capable
    assert [term.contraction for term in tensor.tensor_terms] == ["isotropic", "rank2"]
    rows = tensor.tensor_impulse(DELAYS[1:], 1e25)
    a, b = TENSOR.h_r(DELAYS[1:], "a"), TENSOR.h_r(DELAYS[1:], "b")
    np.testing.assert_allclose(
        rows[0] + rows[1],
        2.0 * TENSOR.raman_fraction * (a + b),
        rtol=0,
        atol=float(1e-15 * np.max(abs(rows[0]))),
    )
    grid = EnvelopeGrid.from_limits(1e-3, 1030e-9, (700e-9, 2e-6), 2e-12)
    compiled = raman_coefficients(
        tensor, None, grid, Constant(1.0), n_pol=2, span=1e-3, tol=1e-12
    )
    assert compiled is not None
    np.testing.assert_array_equal(compiled.tensor_weights, [[1.0, 0.0], [0.25, 0.75]])
    with pytest.raises(TranslationError, match=r"silica_bundle/default:parallel"):
        raman_coefficients(
            scalar, None, grid, Constant(1.0), n_pol=2, span=1e-3, tol=1e-12
        )


def pulse(t, centre, width):
    return np.exp(-(((t - centre) / width) ** 2))


def test_copolarised_and_cross_polarised_probes_after_a_pump():
    # A pump along x, then a weak probe that does not overlap it: the probe's
    # polarisation is its own field times the pump's convolved history, through
    # a + b when it is parallel and a alone when it is crossed (Hellwarth's form).
    response = bundle_response(TENSOR, fraction=None, thg=False, tensor=True)
    n_to, dt = 4096, 2e-16
    t = np.arange(n_to) * dt
    pump = 1e9 * pulse(t, 60e-15, 8e-15)
    probe = 1e3 * pulse(t, 300e-15, 8e-15)
    a = TENSOR.h_r(t, "a")
    b = TENSOR.h_r(t, "b")
    # The same endpoint rule as the sampled kernels: half the limit at zero delay.
    a[0] = 0.0
    b[0] = 0.5 * 2.0 * 0.21 / 96e-15
    drive = np.abs(pump) ** 2 / 2

    def history(kernel):
        return np.convolve(kernel, drive)[:n_to] * dt

    window = slice(int(250e-15 / dt), int(350e-15 / dt))
    for field, kernel, row in (
        (np.stack([pump + probe, np.zeros(n_to)]), a + b, 0),
        (np.stack([pump, probe]), a, 1),
    ):
        polarisation = response.evaluate(
            field.astype(complex), 1.0, kind="envelope", dt=dt
        )
        expected = EPS0 * probe[window] * history(kernel)[window]
        measured = polarisation[row, window] - (
            EPS0 * pump[window] * history(a + b)[window] if row == 0 else 0.0
        )
        # The probe's own drive is 1e-12 of the pump's, and the doubled-grid
        # transform product against np.convolve differs by roundings of the sum.
        np.testing.assert_allclose(
            measured, expected, rtol=0, atol=float(1e-9 * np.max(abs(expected)))
        )


def test_the_tensor_response_takes_optmats_parts_and_fraction():
    from_optmat = SilicaTensorResponse(parameters=TENSOR)
    # The default is optmat's tensor bundle itself since SEP 3, Phase 4b.
    assert SilicaTensorResponse().parameters == TENSOR
    # Against Lin and Agrawal's 2006 parameters supplied as a caller's data (solprop's
    # former registry at dc4ca0d): the same response by an independent route.
    declared = SilicaTensorResponse(parameters=FORMER_SILICA_TENSOR)
    assert from_optmat.fraction == TENSOR.raman_fraction == declared.fraction
    kernels = from_optmat.kernels(DELAYS)
    # The same published response (Lin and Agrawal 2006), the same endpoint rule.
    np.testing.assert_allclose(
        kernels,
        declared.kernels(DELAYS),
        rtol=0,
        atol=float(4e-16 * np.max(abs(kernels))),
    )
    np.testing.assert_allclose(
        from_optmat.hellwarth(DELAYS),
        declared.hellwarth(DELAYS),
        rtol=0,
        atol=float(4e-16 * np.max(abs(kernels))),
    )
    assert "silica_bundle/tensor" in repr(from_optmat)
    rows = dict(from_optmat._describe()[1])
    assert rows["bundle"] == "silica_bundle/tensor (lin_2006)"
    with pytest.raises(MaterialError, match=r"use SilicaTensorResponse\.hellwarth"):
        from_optmat.shapes(DELAYS)


def test_a_tensor_response_of_the_copolarised_bundle_is_refused_by_name():
    with pytest.raises(
        MaterialError, match=r"silica_bundle/default \(hollenbeck_2002\)"
    ):
        SilicaTensorResponse(parameters=DEFAULT)

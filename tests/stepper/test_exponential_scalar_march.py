# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 16 section 7.1: long scalar marches of the PEC recurrence in both formulations.

Exact starting values and closed-form solutions isolate the recurrence's own error
(``tests/exponential_scalar.py``). The cases are the section's: nonlinear feedback with
loss on a constant operator; a driven bin with ``|Im D| h`` from 0.6 to 10; varying
operators over at least a thousand steps, source-free and driven, with relative
variation of ``D`` from 1e-3 to one half and loss, comparing the transported and
untransported compensations with the uncompensated chord as a control; and a weak bin
whose source rotates with its own free phase, as a noise-linear term does. SEP 35's
carrier frame adds a varying carrier and a source that cancels the operator's
variation, so that the source's and the compensation's defects cancel.
"""

import math

import numpy as np
import pytest

from exponential_scalar import ScalarProblem, march

pytestmark = pytest.mark.medium


def kerr(alpha: float, omega: float, gamma: float) -> ScalarProblem:
    """Return ``u' = (-alpha + i omega) u + i gamma |u|^2 u``, ``u(0) = 1``.

    Exact: ``|u|^2 = exp(-2 alpha z)``, so the nonlinear phase is
    ``gamma (1 - exp(-2 alpha z)) / (2 alpha)``.
    """
    rate = complex(-alpha, omega)

    def exact(z: float) -> complex:
        return np.exp(
            rate * z + 1j * gamma * (1 - math.exp(-2 * alpha * z)) / (2 * alpha)
        )

    return ScalarProblem(
        lambda z: rate,
        lambda z: rate * z,
        lambda z, u: 1j * gamma * abs(u) ** 2 * u,
        exact,
    )


def chirped(w0: float, a: float, alpha: float, kind: str) -> ScalarProblem:
    """Return a bin with ``D(z) = i (w0 + a z) - alpha`` and a manufactured source.

    ``"free"``: no source, ``u = exp(Phi)``. ``"driven"``: the state follows a smooth
    lab-frame amplitude ``A(z) = (1 + sin(3 z) / 2) exp(2 i z)``, ``F = A' - D A``.
    ``"rotating"``: ``u = exp(Phi) (1 + sin(3 z) / 2)``, whose source
    ``F = exp(Phi) 3 cos(3 z) / 2`` rotates with the bin's own free phase.
    """

    def rate(z: float) -> complex:
        return 1j * (w0 + a * z) - alpha

    def phase(z: float) -> complex:
        return 1j * (w0 * z + a * z * z / 2) - alpha * z

    if kind == "free":
        return ScalarProblem(rate, phase, lambda z, u: 0j, lambda z: np.exp(phase(z)))
    if kind == "driven":

        def amplitude(z: float) -> complex:
            return (1 + 0.5 * math.sin(3 * z)) * np.exp(2j * z)

        def slope(z: float) -> complex:
            return (1.5 * math.cos(3 * z) + 2j * (1 + 0.5 * math.sin(3 * z))) * np.exp(
                2j * z
            )

        return ScalarProblem(
            rate, phase, lambda z, u: slope(z) - rate(z) * amplitude(z), amplitude
        )
    return ScalarProblem(
        rate,
        phase,
        lambda z, u: np.exp(phase(z)) * 1.5 * math.cos(3 * z),
        lambda z: np.exp(phase(z)) * (1 + 0.5 * math.sin(3 * z)),
    )


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_pec_converges_at_one_order_above_its_predictor(formulation, order):
    # Nonlinear feedback with loss, its self-rotation resolved (|Im D| h <= 0.4).
    problem = kerr(0.5, 50.0, 3.0)
    # The interaction formulation's error constant is up to 1e3 smaller here, so its
    # ladder is coarser, to stay above the round-off floor at order six.
    steps = (80, 160, 320) if formulation == "interaction" else (250, 500, 1000)
    errors = [march(problem, 1.0, n, order, formulation) for n in steps]
    observed = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    # Three levels and an order gate: PEC q/(q+1) is order q + 1 globally (its
    # predicted states advance by Adams-Bashforth q + 1). Measured 1.95-6.14 against
    # 2-6; the margins admit a pre-asymptotic first level and the super-convergence
    # of a smooth problem (5.59 at q = 4), and the upper one would still catch a
    # poisoned reference (docs/dev/stepper.md section 8).
    assert all(order + 1 - 0.25 <= one <= order + 1 + 1.0 for one in observed), observed
    assert errors[-1] > 1e3 * np.finfo(float).eps


@pytest.mark.parametrize("order", [1, 3, 5])
def test_exponential_formulation_resolves_a_driven_bin_at_large_linear_phase(order):
    # A constant operator, |Im D| h from 10 down to 0.6, a state that follows its
    # smooth lab-frame drive: exact integration against the flow leaves only the
    # drive's interpolation error, while the transported history rotates by |Im D| h
    # per step and the interaction formulation fails (SEP 15 section 4).
    problem = chirped(2000.0, 0.0, 0.5, "driven")
    for steps in (200, 400, 800, 3200):
        exponential = march(problem, 1.0, steps, order, "exponential")
        interaction = march(problem, 1.0, steps, order, "interaction")
        # The drive's derivatives are O(1) per metre: the error at h = 1/200 is
        # bounded by the predictor's local defect, 1e-5 at order one.
        assert exponential < 2e-5
        assert interaction > 1e3 * exponential


VARIATIONS = {
    # (w0, a): relative variation of D over the unit run a / w0.
    "variation 1e-3": (2000.0, 2.0),
    "variation 1e-2": (2000.0, 20.0),
    "variation 1/2": (200.0, 100.0),
}


@pytest.mark.parametrize(("w0", "a"), VARIATIONS.values(), ids=VARIATIONS.keys())
@pytest.mark.parametrize("alpha", [0.0, 0.5], ids=["lossless", "lossy"])
def test_source_free_varying_operator(w0, a, alpha):
    problem = chirped(w0, a, alpha, "free")
    errors = {}
    for steps in (1000, 2000, 4000):
        for treatment in ("transported", "untransported", "chord"):
            errors[treatment, steps] = march(
                problem, 1.0, steps, 4, "exponential", treatment
            )
    for steps in (1000, 2000, 4000):
        # The chord is the exact homogeneous flow: round-off only, accumulated over
        # the steps from phases of size w0 (w0 * eps per step).
        assert errors["chord", steps] < steps * w0 * 1e-15
        # The transported compensation is smooth in the reference flow's frame and
        # converges; the untransported one interpolates a sequence rotating with the
        # bin's free phase (SEP 16 section 4.2).
        assert errors["transported", steps] < 1e-6
        assert errors["untransported", steps] > 1e2 * errors["transported", steps]


@pytest.mark.parametrize(("w0", "a"), VARIATIONS.values(), ids=VARIATIONS.keys())
def test_driven_varying_operator(w0, a):
    problem = chirped(w0, a, 0.5, "driven")
    for steps in (1000, 2000, 4000):
        transported = march(problem, 1.0, steps, 4, "exponential", "transported")
        untransported = march(problem, 1.0, steps, 4, "exponential", "untransported")
        chord = march(problem, 1.0, steps, 4, "exponential", "chord")
        # A driven bin's state is smooth in the lab frame, so here the untransported
        # compensation is the smooth one and is accurate to round-off; transporting it
        # makes it rotate. The transported compensation still stays within an order
        # of magnitude of the uncompensated chord, as section 4.2 states, and beats
        # it once |Im D| h falls below about one.
        assert untransported < 1e-11
        assert transported < 10 * chord
        if w0 / steps <= 0.5:
            assert transported < chord


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
def test_a_source_rotating_with_the_free_phase(formulation):
    # A noise-linear source in a weak bin rotates with the bin's own phase: the
    # transported history is smooth (interaction), the untransported one is not, so
    # the exponential formulation converges only once |Im D| h is small (section 4.4).
    problem = chirped(2000.0, 20.0, 0.5, "rotating")
    errors = [march(problem, 1.0, n, 4, formulation) for n in (1600, 3200, 6400)]
    if formulation == "interaction":
        # The transported source is exactly the polynomial's: round-off only.
        assert max(errors) < 1e-9
    else:
        observed = math.log2(errors[1] / errors[2])
        # Fifth order once resolved (w0 h = 1.25 to 0.31): measured 4.9.
        assert 4.4 <= observed <= 6.5
        assert errors[0] > 1e3 * max(
            march(problem, 1.0, 1600, 4, "interaction"), np.finfo(float).eps
        )


@pytest.mark.parametrize("compensation", ["transported", "untransported"])
@pytest.mark.parametrize("order", [2, 4])
def test_the_carrier_frame_follows_a_varying_carrier(compensation, order):
    # SEP 35 section 4.5: a source rotating with a varying carrier, D = i (w0 + a z),
    # leaves the chord's shift a residual phase a h^2 theta (1 - theta) / 2, which the
    # polynomial resolves. w0 h runs from 20 to 5: far beyond what the unshifted
    # history resolves (measured 2e-2 to 6e-2 unshifted, 840 to 2e8 times the
    # shifted error), and three levels show the recurrence's order.
    problem = chirped(2000.0, 20.0, 0.5, "rotating")
    steps = (100, 200, 400)
    shifted = [
        march(problem, 1.0, n, order, "exponential", compensation, shift=True)
        for n in steps
    ]
    unshifted = [
        march(problem, 1.0, n, order, "exponential", compensation) for n in steps
    ]
    observed = [math.log2(shifted[i] / shifted[i + 1]) for i in range(2)]
    # PEC q/(q+1) is order q + 1 globally; the same gate as the constant operator's.
    assert all(order + 1 - 0.25 <= one <= order + 1 + 1.0 for one in observed), observed
    assert all(error > 1e2 * one for error, one in zip(unshifted, shifted, strict=True))
    assert shifted[-1] > 1e3 * np.finfo(float).eps


def test_source_and_compensation_defects_cancel_under_the_carrier_frame():
    # F(z, u) = -(D(z) - D(0)) u cancels the operator's variation: u = exp(D(0) z). The
    # source and the transported compensation are each of order a z |u| and their
    # defects cancel in the sum (SEP 35 section 4.5); the frame changes the source's
    # samples and so how they cancel, not the result's order.
    w0, a, alpha = 2000.0, 100.0, 0.5
    start = 1j * w0 - alpha
    problem = ScalarProblem(
        lambda z: 1j * (w0 + a * z) - alpha,
        lambda z: 1j * (w0 * z + a * z * z / 2) - alpha * z,
        lambda z, u: -1j * a * z * u,
        lambda z: np.exp(start * z),
    )
    steps = (400, 800, 1600)
    errors = [march(problem, 1.0, n, 3, "exponential", shift=True) for n in steps]
    observed = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(4 - 0.25 <= one <= 4 + 1.0 for one in observed), observed
    # The unshifted march converges too (its compensation is the smooth part); the
    # frame stays within an order of magnitude of it.
    unshifted = march(problem, 1.0, steps[-1], 3, "exponential")
    assert errors[-1] < 10 * unshifted

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The silica tensor Raman model: its kernels, its contraction and its limits.

What is tested here: the two seam kernels against their declared integrals and
against an independent quadrature; the scalar limit against the existing scalar Raman
machinery fed the same kernels; the cross-polarised delay a tensor response has and a
scalar one cannot; rotation covariance in three dimensions; the three-component
history and its six independent entries; the delta, zero and long-delay limits; the
Stokes sign in propagation; and that nothing wraps around.

The response is the caller's Lin--Agrawal route (``SilicaTensorResponse`` of a
:class:`LinAgrawalRaman`), fed the parameters solprop's former registry declared
(dc4ca0d; ``former_raman.FORMER_SILICA_TENSOR``): the tensor structure is Lin and
Agrawal, J. Opt. Soc. Am. B 20, 1616 (2003), Eqs. (2.2) and (2.5), and the parameters
are Opt. Lett. 31, 3086 (2006). The default, optmat's tensor bundle of the same shape,
is tested in test_silica_bundle_raman.py; nothing here depends on the parameters'
values except the two tests that name them.

**Every bound is written before the run.** A causal convolution on a doubled grid and
a four-index contraction are exact arithmetic, so the agreements below are rounding
statements whose budget is the reduction length; the two quadrature comparisons carry
their own discretisation bound, stated where they are made.
"""

from itertools import pairwise

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from former_raman import FORMER_SILICA_TENSOR
from solprop.constants import EPS0
from solprop.exceptions import GridError, MaterialError
from solprop.nonlinear.gnlse import kerr_susceptibility, raman_susceptibility
from solprop.nonlinear.raman import (
    SILICA_TENSOR_AMPLITUDE,
    SILICA_TENSOR_INTEGRALS,
    RamanResponse,
    RamanTensorTerm,
    SilicaTensorResponse,
    normalised_damped_oscillator,
    response_for,
    undamped_dephasing,
)
from solprop.physdata import LinAgrawalRaman

CHI3 = 2.2e-22
"""A silica-like absolute susceptibility, m²/V², declared by these tests."""

DT = 1e-15
"""Fine sample spacing of the delay axis, s: 12 samples per tau1."""

SAMPLES = 2048
"""Fine samples: 2 ps, about twenty times the longest declared time constant."""

ROUNDING = 1e-12
"""Budget for an identity that is arithmetic, not physics.

Two real transforms of length ``2 * SAMPLES`` and a contraction over three
components: a few tens of thousands of products of one magnitude, which at
2.2e-16 each is of order 1e-11 relative in the worst case and far less in
practice. The same number and the same reason as the general route's own
rounding budget.
"""


@pytest.fixture(scope="module")
def response():
    """The Lin--Agrawal silica tensor response, of supplied parameters."""
    return SilicaTensorResponse(parameters=FORMER_SILICA_TENSOR)


@pytest.fixture(scope="module")
def delays():
    """The causal delay axis the kernels are sampled on."""
    return np.arange(SAMPLES) * DT


def gaussian_field(components, width=25e-15, amplitude=1e9, centre=400):
    """Return a real three-component Gaussian pulse on the fine time axis.

    Parameters
    ----------
    components : sequence of float
        The Cartesian weights of the (linear) polarisation.
    width : float, optional
        Half-width of the Gaussian envelope, s.
    amplitude : float, optional
        Peak field, V/m.
    centre : int, optional
        Sample the pulse is centred on.

    Returns
    -------
    numpy.ndarray
        Shape ``(3, SAMPLES)``, V/m, with a carrier so that the drive has a beat.
    """
    t = (np.arange(SAMPLES) - centre) * DT
    carrier = np.cos(2 * np.pi * 3.59e14 * t)
    profile = amplitude * np.exp(-((t / width) ** 2)) * carrier
    return np.stack([weight * profile for weight in components])


# --------------------------------------------------------------- the declared data


@pytest.mark.parametrize(
    "change,message",
    [
        ({"tau1": 0.0}, "positive, finite time"),
        ({"tau2": float("nan")}, "positive, finite time"),
        ({"tau_b": -1e-15}, "positive, finite time"),
        ({"isotropic_weight": -0.1, "boson_weight": 1.06}, "non-negative weights"),
        ({"boson_weight": 0.3}, "must sum to one"),
        ({"fraction": 0.0}, r"\(0, 1\]"),
        ({"fraction": 1.5}, r"\(0, 1\]"),
    ],
)
def test_the_parameter_record_refuses_what_it_cannot_carry(change, message):
    """A weight set that does not share one fraction is not a response."""
    declared = {
        "tau1": 12.2e-15,
        "tau2": 32.0e-15,
        "tau_b": 96.0e-15,
        "isotropic_weight": 0.75,
        "boson_weight": 0.21,
        "anisotropic_weight": 0.04,
        "fraction": 0.245,
        "citation": "a test",
    }
    with pytest.raises(MaterialError, match=message):
        LinAgrawalRaman(**(declared | change))


# ------------------------------------------------------------------- the kernels


def test_the_two_shapes_integrate_to_one_against_an_independent_quadrature(response):
    """`h_a` and `h_b` are normalised, by adaptive quadrature and not by sampling.

    The bound is SciPy's own: `quad` on a smooth decaying integrand over four
    picoseconds reports an absolute error estimate, and 1e-9 is four orders above
    the largest of them and six below the value being checked.
    """
    for row in (0, 1):
        value, error = quad(
            lambda t, row=row: float(response.shapes(np.array([t, t + DT]))[row, 0]),
            0.0,
            4e-12,
            limit=400,
        )
        assert error < 1e-9
        assert value == pytest.approx(1.0, abs=1e-9)


def test_the_vibrational_shape_is_the_existing_damped_oscillator(response, delays):
    """`h_a` is the normalised damped oscillator this package already builds.

    An independent construction of the same function through a different owner:
    `normalised_damped_oscillator(1, 1/tau1, tau2)` is `K exp(-t/tau2) sin(t/tau1)`
    scaled so that its integral is one, which is exactly the Blow-Wood `h_a` of the
    Lin-Agrawal form.
    """
    parameters = response.parameters
    line = normalised_damped_oscillator(1.0, 1.0 / parameters.tau1, parameters.tau2)
    expected = line.undamped(delays) * line.dephasing.damping(delays, 0.0)
    # Scaled by the peak and not relative, because the two constructions group the
    # same product differently and a sample near a zero of the sine cancels to
    # nothing: a relative bound there measures the cancellation, not the agreement.
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    assert_allclose(
        response.shapes(delays)[0], expected, rtol=0.0, atol=ROUNDING * scale
    )


def test_the_seam_kernels_integrate_to_five_sixths_and_one_sixth(response):
    """The A2 companion's 5/6 and 1/6, and not the paper's own 3/4 and 1/4.

    ``r_trace = a + b/3`` and ``r_rank = 2 b/3`` come from matching
    ``a tr(Q) E + b Q E`` to the trace and ``T(Q) = (3 Q - tr(Q) I)/2`` channels the
    seam carries; because ``h_a`` and ``h_b`` are each normalised, the two shape
    integrals are ``w_a + (w_b + w_ab)/3`` and ``2 (w_b + w_ab)/3`` and sum to one.
    They are 5/6 and 1/6 **because** ``w_a = 0.75``, which is asserted here rather
    than assumed. Adaptive quadrature again, with the same bound.

    The kernels carry ``(3/2) f_R`` on top of the shape -- the field-domain
    amplitude mapping of `SILICA_TENSOR_AMPLITUDE` -- so the quadrature is divided
    by it before the shape integrals are read off.
    """
    parameters = response.parameters
    anisotropic = parameters.boson_weight + parameters.anisotropic_weight
    assert (
        parameters.isotropic_weight + anisotropic / 3.0,
        2.0 * anisotropic / 3.0,
    ) == SILICA_TENSOR_INTEGRALS
    assert parameters.isotropic_weight == 0.75
    assert SILICA_TENSOR_INTEGRALS == (5.0 / 6.0, 1.0 / 6.0)
    amplitude = SILICA_TENSOR_AMPLITUDE * response.fraction
    totals = []
    for row in range(2):
        value, error = quad(
            lambda t, row=row: float(response.kernels(np.array([t, t + DT]))[row, 0]),
            0.0,
            4e-12,
            limit=400,
        )
        assert error < 1e-9
        totals.append(value / amplitude)
    assert_allclose(totals, SILICA_TENSOR_INTEGRALS, rtol=0.0, atol=1e-9)
    # And the two together are the scalar model's own h_R, which integrates to one.
    assert sum(totals) == pytest.approx(1.0, abs=2e-9)


def test_the_executed_kernels_integrate_to_the_shape_integrals(response):
    """The **sampled, weighted** kernels, which is what the seam actually executes.

    The endpoint treatment is a choice the implementation must make and record.
    ``h_b(0+) = 2/tau_b`` is not zero, so a causal kernel sampled on ``arange(n) dt``
    and integrated by the rectangle-rule transform product misses ``dt h(0+)/2``. The
    declared choice is the **trapezoid** -- the sample at the jump carries half the
    right-hand limit -- and this test measures the executed sum and not the continuous
    integral.

    **Two bounds, both written before the run.** On the finest grid the relative
    departure from `SILICA_TENSOR_INTEGRALS` must be below **1e-4**: what is left
    after the endpoint is Euler-Maclaurin's ``dt^2 h'(0+)/12``, and with
    ``h'(0+) = w_a (tau1^2 + tau2^2)/(tau1 tau2)^2 + ...`` of order 5.9e27 1/s^2 on
    the trace channel that is 3.0e-5 absolute at ``dt = 0.25`` fs, 3.7e-5 of 5/6.
    And halving the spacing must divide the error by **at least 3.5**: second
    order, asserted with the margin an asymptotic rate needs.

    Before the endpoint fix the executed integrals were first-order biased --
    measured 0.833256 / 0.832847 / 0.832117 (trace) and
    0.166521 / 0.165934 / 0.165195 (rank two) at dt = 0.1 / 0.5 / 1 fs, against
    5/6 and 1/6.
    """
    amplitude = SILICA_TENSOR_AMPLITUDE * response.fraction
    errors = []
    for samples in (4000, 8000, 16000):
        dt = 4e-12 / samples
        totals = response.kernels(np.arange(samples) * dt).sum(axis=1) * dt / amplitude
        errors.append(np.abs(totals - np.array(SILICA_TENSOR_INTEGRALS)))
    assert_allclose(
        errors[-1] / np.array(SILICA_TENSOR_INTEGRALS), 0.0, rtol=0.0, atol=1e-4
    )
    for coarse, fine in pairwise(errors):
        assert np.all(coarse / fine > 3.5)


def test_the_two_shares_add_to_one_chi3_on_the_fundamental(response, delays):
    """The seam's analogue of `gnlse`'s own Kerr/Raman identity.

    ``f_R`` and ``n2`` are **envelope** quantities: ``f_R`` is the share of the
    effective nonlinearity on the fundamental the delayed term carries. In the
    field domain the delayed polarisation's drive is the cycle average
    ``|A|^2/2``, so a kernel of integral ``X`` contributes ``(1/2) X |A|^2 A``
    against the instantaneous cubic's ``(3/4) chi3 |A|^2 A`` -- it acts like a Kerr
    term of ``(2/3) X``. Carrying the share ``f_R`` of one ``chi3`` therefore needs
    ``X = (3/2) f_R chi3``, which is `SILICA_TENSOR_AMPLITUDE`, and then

    ``(2/3) X + (1 - f_R) chi3 == chi3`` .

    This is exactly `gnlse.raman_susceptibility`'s own identity
    ``(2/3) f_R raman_susceptibility(n2) + kerr_susceptibility(n2, f_R) ==
    kerr_susceptibility(n2)`` transported to this seam, and the first assertion is
    that the two amplitudes are the **same number**: `raman_susceptibility` is
    ``2 n2 eps0 c`` and ``(3/2) kerr_susceptibility(n2)`` is the same thing.
    Both are exact arithmetic on floats, so the bound is 1e-15 relative.

    The third assertion closes it on the **executed** kernels rather than on the
    declared amplitude, with the trapezoid quadrature's own bound: at
    ``dt = 1`` fs the shape integrals are 5.9e-4 relative low, which carries
    ``(2/3) (3/2) f_R = 0.245`` of itself into the identity, so 1e-3 absolute.
    Without the ``3/2`` this sum would be ``1 - f_R/3 = 0.9183``, which is the
    8.2 % the declared ``n2`` was low by.
    """
    n2 = 2.6e-20
    chi3 = kerr_susceptibility(n2)
    amplitude = SILICA_TENSOR_AMPLITUDE * response.fraction
    assert amplitude * chi3 == pytest.approx(
        response.fraction * raman_susceptibility(n2), rel=1e-15
    )
    assert (2.0 / 3.0) * amplitude * chi3 + response.instantaneous_share * chi3 == (
        pytest.approx(chi3, rel=1e-15)
    )
    executed = float(response.kernels(delays).sum(axis=0).sum()) * DT
    assert (2.0 / 3.0) * executed + response.instantaneous_share == pytest.approx(
        1.0, abs=1e-3
    )


def test_the_kernels_are_causal_and_die_inside_the_window(response, delays):
    """Zero before zero delay, half the limit at it, negligible by the window's end.

    The last part is what makes the doubled-grid convolution enough: the response
    must have decayed long before the padding ends, or a linear convolution on this
    window would not be the infinite one.

    The sample **at** zero delay is the trapezoid endpoint and is not zero: it is
    half the kernel's right-hand limit there, ``h_a(0+) = 0`` and
    ``h_b(0+) = 2/tau_b``. That costs no causality -- it multiplies ``Q(t)`` itself
    and no earlier sample -- and every sample at a strictly negative delay is still
    identically zero, which is the statement that matters.
    """
    kernels = response.kernels(delays)
    amplitude = SILICA_TENSOR_AMPLITUDE * response.fraction
    boson = response.parameters.boson_weight * 2.0 / response.parameters.tau_b
    endpoint = amplitude * 0.5 * np.array([boson / 3.0, 2.0 * boson / 3.0])
    # `ROUNDING`, not exactly: the two group the same four factors differently.
    assert_allclose(kernels[:, 0], endpoint, rtol=ROUNDING, atol=0.0)
    assert np.all(response.kernels(-delays[1:])[:, :] == 0.0)
    peak = np.max(np.abs(kernels), axis=1)
    assert np.all(peak > 0.0)
    # The slowest term is the boson peak, exp(-t/96 fs) times a linear ramp, so at
    # 2 ps the tail is of order exp(-20.8) * t/tau_b -- about 2e-8 of the peak. The
    # declared bound is 1e-7; measured 1.8e-10 (trace) and 4.0e-9 (rank two).
    tail = np.abs(kernels[:, -1]) / peak
    assert np.all(tail < 1e-7)


def test_the_two_channel_tensors_are_the_declared_contraction(response):
    """`D[trace]` and `D[rank]`, against a literal four-index sum.

    Written out index by index rather than as the same `einsum` the class uses, so
    the two constructions share nothing but the definition.
    """
    trace, rank = response.tensors()
    expected_trace = np.zeros((3, 3, 3, 3))
    expected_rank = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):  # noqa: E741 - the tensor's own index name
                    delta_ij = float(i == j)
                    delta_kl = float(k == l)
                    expected_trace[i, j, k, l] = delta_ij * delta_kl
                    expected_rank[i, j, k, l] = (
                        3.0
                        * (
                            float(i == k) * float(j == l)
                            + float(i == l) * float(j == k)
                        )
                        - 2.0 * delta_ij * delta_kl
                    ) / 4.0
    assert_allclose(trace, expected_trace, rtol=0.0, atol=0.0)
    assert_allclose(rank, expected_rank, rtol=0.0, atol=0.0)
    # Symmetric in the last two indices, which is what makes six histories enough.
    assert_allclose(rank, np.swapaxes(rank, 2, 3), rtol=0.0, atol=0.0)
    # And the rank-two channel is traceless in (i, j) for a traceless history.
    history = np.array([[2.0, 1.0, 0.0], [1.0, -2.0, 0.5], [0.0, 0.5, 0.0]])
    contracted = np.einsum("ijkl,kl->ij", rank, history)
    assert float(np.trace(contracted)) == pytest.approx(0.0, abs=1e-14)


# ------------------------------------------------------------------ the responses


def scalar_oracle(response, field, chi3, *, thg=True):
    """Return the scalar-limit polarisation through the existing Raman machinery.

    An independent path to the same number: the two seam kernels are declared as
    ordinary scalar `RamanModel`s of the existing `RamanResponse`, which convolves
    and multiplies by the field in its own code, on its own doubled grid.
    """
    delays = np.arange(field.size) * DT
    kernels = response.kernels(delays)

    class _Sampled:
        """A scalar Raman model that is one row of the tensor response's kernels."""

        def __init__(self, values):
            self.values = values
            # The package's own undamped dephasing, so that this stands in for a
            # `RamanModel` structurally as well as by duck typing.
            self.dephasing = undamped_dephasing()

        def undamped(self, t):
            """Return the sampled kernel at these delays.

            Parameters
            ----------
            t : numpy.ndarray
                Delays in s, on the same axis the kernel was sampled on.

            Returns
            -------
            numpy.ndarray
                The kernel, 1/s.
            """
            assert np.allclose(t, delays)
            return self.values

    total = np.zeros_like(field)
    for row in range(2):
        scalar = RamanResponse(
            tensor_terms=(RamanTensorTerm(_Sampled(kernels[row]), "isotropic"),),
            thg=thg,
        )
        total = (
            total + scalar.evaluate(field[None, :], 1.0, kind="field", dt=DT)[0] / EPS0
        )
    return EPS0 * chi3 * total


def test_the_scalar_limit_is_the_existing_scalar_raman_arithmetic(response):
    """One fixed linear polarisation reduces the tensor to the fibre model.

    The scalar-limit check. The two channels add to
    ``E(t) int f_R h_R(tau) E^2(t - tau) dtau``, which is what the existing
    `RamanResponse` computes from the same kernels through its own convolution and
    its own multiplication by the field. The bound is `ROUNDING`: the same integral,
    grouped differently.
    """
    field = gaussian_field([1.0, 0.0, 0.0])
    polarisation = response.evaluate(field, CHI3, kind="field", dt=DT)
    expected = scalar_oracle(response, field[0], CHI3)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    assert_allclose(polarisation[0], expected, rtol=0.0, atol=ROUNDING * scale)
    # The two transverse partners of a linear polarisation stay empty: an isotropic
    # medium driven along x radiates along x.
    assert_allclose(polarisation[1:], 0.0, rtol=0.0, atol=ROUNDING * scale)


def test_a_one_component_field_is_the_three_component_one_restricted(response):
    """The tensor is cut to the components it is given, not broadcast into them."""
    field = gaussian_field([1.0, 0.0, 0.0])
    three = response.evaluate(field, CHI3, kind="field", dt=DT)
    one = response.evaluate(field[:1], CHI3, kind="field", dt=DT)
    scale = float(np.max(np.abs(three)))
    assert_allclose(one[0], three[0], rtol=0.0, atol=ROUNDING * scale)


def test_a_cross_polarised_probe_sees_three_quarters_of_the_response(response):
    """The delay a scalar response cannot have.

    A strong pump along x, and a weak probe along y that arrives **after** it. In
    Lin and Agrawal's Eq. (2.2) the probe's delayed polarisation is then the
    isotropic term alone,

    ``P_y(t) = eps0 chi3 E_y(t) int a(tau) (E.E)(t - tau) dtau`` ,

    because the dyadic term in ``b`` needs the two to overlap. In the seam's two
    channels that is ``r_trace - r_rank/2 = a``, so a cross-polarised probe sees
    **three quarters** of what a co-polarised one sees -- ``int a = 3/4`` against
    ``int (a + b) = 1``. A scalar Raman model gives it the whole ``h_R`` and is
    wrong by that factor of 4/3, which is the measurement this test makes.
    """
    pump = gaussian_field([1.0, 0.0, 0.0], centre=400)
    probe = 1e-3 * gaussian_field([0.0, 1.0, 0.0], centre=1100)
    pump_alone = response.evaluate(pump, CHI3, kind="field", dt=DT)
    driven = (
        response.evaluate(pump + probe, CHI3, kind="field", dt=DT)
        - pump_alone
        - response.evaluate(probe, CHI3, kind="field", dt=DT)
    )[1]
    # The cross term is read out of a difference of three numbers of the pump's own
    # size, so the floor is that cancellation and not the cross term's own scale:
    # `ROUNDING` times the pump's polarisation peak, declared before the run.
    cancellation = ROUNDING * float(np.max(np.abs(pump_alone)))
    trace, rank = response.kernels(np.arange(SAMPLES) * DT)
    isotropic = trace - 0.5 * rank
    history = np.fft.irfft(
        np.fft.rfft(isotropic, n=2 * SAMPLES)
        * DT
        * np.fft.rfft(pump[0] ** 2, n=2 * SAMPLES),
        n=2 * SAMPLES,
    )[:SAMPLES]
    expected = EPS0 * CHI3 * history * probe[1]
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    assert_allclose(driven, expected, rtol=0.0, atol=cancellation)
    # And the peak agrees to a millionth, which the cancellation floor does not
    # reach: the floor is 2e-18 and the signal is 9e-20 at its largest, so the
    # waveform statement above is the weaker of the two and this is the sharp one.
    assert float(np.max(np.abs(driven))) == pytest.approx(scale, rel=1e-6)
    # And `r_trace - r_rank/2` **is** the paper's own a(tau) = w_a h_a, exactly:
    # the boson peak cancels out of it (0.07 - 0.14/2 = 0), which is why a
    # cross-polarised probe sees the 32 fs vibrational response alone and not the
    # 96 fs one. Its integral is w_a = 3/4 of the whole response's one.
    parameters = response.parameters
    shape_a = response.shapes(np.arange(SAMPLES) * DT)[0]
    paper = (
        SILICA_TENSOR_AMPLITUDE
        * response.fraction
        * parameters.isotropic_weight
        * shape_a
    )
    assert parameters.isotropic_weight == 0.75
    peak = float(np.max(np.abs(paper)))
    assert_allclose(isotropic, paper, rtol=0.0, atol=ROUNDING * peak)


def test_the_response_is_covariant_under_a_three_dimensional_rotation(response):
    """Rotate the field, rotate the answer: an isotropic medium has no axes.

    The rotation-covariance check, in three dimensions and not in the
    transverse plane alone: the longitudinal component of a guided mode is what
    makes that distinction matter.
    """
    field = gaussian_field([0.6, 0.8, 0.0]) + gaussian_field(
        [0.0, 0.3, 0.9], centre=700
    )
    angle = 0.7
    axis = np.array([1.0, 2.0, -0.5])
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross
        + (1 - np.cos(angle)) * np.outer(axis, axis)
    )
    assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    direct = response.evaluate(rotation @ field, CHI3, kind="field", dt=DT)
    rotated = rotation @ response.evaluate(field, CHI3, kind="field", dt=DT)
    scale = float(np.max(np.abs(rotated)))
    assert scale > 0.0
    assert_allclose(direct, rotated, rtol=0.0, atol=ROUNDING * scale)


def test_the_off_diagonal_history_entries_are_load_bearing(response):
    """Six entries carry the symmetric history, and all six are needed.

    The drive `Q_kl = E_k E_l` is symmetric, so its upper triangle is the whole of
    it -- but a test that only rebuilds the triangle it packed itself asserts
    nothing about the code. What is asserted here is the part that is the model's:
    the three off-diagonal entries genuinely drive the answer, so dropping them
    changes the polarisation of a field whose three components are all excited.
    Only the **rank-two** channel sees them -- the trace channel contracts
    ``H_kk`` and nothing else -- so the bound is a declared **5 %** of the peak
    rather than an order-one fraction of it; measured 8.2 %.

    The **production** packing and its inverse are asserted where they live:
    `test_the_six_stored_entries_rebuild_the_symmetric_history` and
    `test_the_delayed_pass_is_the_reference_tensor_response` in
    ``tests/backends/test_general_modal.py``, which run `expand_pair_history` and
    the backend's own six-history pass against this class's 3x3 contraction.
    """
    field = gaussian_field([0.5, -0.7, 0.4]) + gaussian_field([0.2, 0.1, -0.8], 40e-15)
    drive = np.einsum("kt,lt->klt", field, field)
    assert_allclose(drive, np.swapaxes(drive, 0, 1), rtol=0.0, atol=0.0)
    polarisation = response.evaluate(field, CHI3, kind="field", dt=DT)
    assert np.all(np.max(np.abs(polarisation), axis=1) > 0.0)
    # Blank the three off-diagonal pairs of the drive and evaluate the same
    # contraction by hand: if they carried nothing the two would agree.
    kernels = response.kernels(np.arange(SAMPLES) * DT)
    tensors = response.tensors()

    def contract(quadratic):
        """Return the polarisation this drive produces, in C/m^2."""
        total = np.zeros(field.shape)
        for kernel, tensor in zip(kernels, tensors, strict=True):
            history = np.fft.irfft(
                np.fft.rfft(kernel, n=2 * SAMPLES)
                * DT
                * np.fft.rfft(quadratic, n=2 * SAMPLES, axis=-1),
                n=2 * SAMPLES,
                axis=-1,
            )[..., :SAMPLES]
            total = total + np.einsum("ijkl,klt,jt->it", tensor, history, field)
        return EPS0 * CHI3 * total

    scale = float(np.max(np.abs(polarisation)))
    assert_allclose(contract(drive), polarisation, rtol=0.0, atol=ROUNDING * scale)
    diagonal = drive * np.eye(3)[:, :, None]
    assert float(np.max(np.abs(contract(diagonal) - polarisation))) > 0.05 * scale


# --------------------------------------------------------------------- the limits


def test_a_delta_like_drive_returns_the_kernel_itself(response):
    """The impulse response, which is the kernel by definition.

    A pump of one sample at index zero and a weak constant probe after it, read
    through the **cross** term: the drive of the pair is exactly the sum of the two
    separate drives, because the pump and the probe never overlap in time, so
    subtracting the two single-field answers leaves the pump's history acting on
    the probe -- ``A^2 dt (h_trace + h_rank)(t) probe(t)``, with the discrete delta
    of weight ``dt``.
    """
    pump = np.zeros((3, SAMPLES))
    pump[0, 0] = 1e9
    probe = np.zeros((3, SAMPLES))
    probe[0, 1:] = 1e3
    both = response.evaluate(pump + probe, CHI3, kind="field", dt=DT)
    alone = response.evaluate(pump, CHI3, kind="field", dt=DT)
    weak = response.evaluate(probe, CHI3, kind="field", dt=DT)
    driven = (both - alone - weak)[0]
    kernels = response.kernels(np.arange(SAMPLES) * DT)
    expected = (
        EPS0 * CHI3 * float(pump[0, 0]) ** 2 * DT * kernels.sum(axis=0) * probe[0]
    )
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    assert_allclose(driven[1:], expected[1:], rtol=0.0, atol=ROUNDING * scale)


def test_a_zero_delay_and_a_long_delay_bracket_the_response(response):
    """Nothing before the drive, and nothing left long after it.

    Causality and decay, on the response itself rather than on the kernel: a pulse
    at sample 400 leaves the polarisation identically zero before it and below a
    part in a million of its peak by the end of a 2 ps window.
    """
    field = gaussian_field([1.0, 0.0, 0.0], centre=400)
    polarisation = response.evaluate(field, CHI3, kind="field", dt=DT)
    peak = float(np.max(np.abs(polarisation)))
    assert peak > 0.0
    # Before the pulse the drive is the Gaussian's own tail, which at eight widths
    # is exp(-64) of its peak; the bound is 1e-12 of the polarisation's peak.
    assert float(np.max(np.abs(polarisation[:, :200]))) < 1e-12 * peak
    assert float(np.max(np.abs(polarisation[:, -200:]))) < 1e-6 * peak


def test_nothing_wraps_around_the_window(response):
    """The doubled grid is what makes the convolution linear, and it is tested.

    A drive placed at the **end** of the window leaves the head untouched. A
    circular convolution on the single grid would wrap its response onto the first
    samples, which is exactly the failure the doubled, zero-padded grid prevents.
    """
    field = gaussian_field([1.0, 0.0, 0.0], centre=SAMPLES - 60, width=10e-15)
    polarisation = response.evaluate(field, CHI3, kind="field", dt=DT)
    peak = float(np.max(np.abs(polarisation)))
    assert peak > 0.0
    head = float(np.max(np.abs(polarisation[:, : SAMPLES - 200])))
    # A circular convolution would put the response's whole tail here, of order
    # 1e-2 of the peak; the bound is 1e-12, and the measured head is the Gaussian's
    # own far tail and nothing else.
    assert head < 1e-12 * peak


def test_the_stokes_side_is_amplified_and_the_anti_stokes_side_is_not(response):
    """The sign that says this is Raman gain and not Raman loss.

    A pump and a weak sideband, one below the pump in frequency and one above. The
    energy the field gains from the medium is ``-int E dP/dt dt``, which is
    ``+int P dE/dt dt`` after one integration by parts over a window the pulse
    vanishes at either end of. It is **positive** for the red sideband and negative
    for the blue one: energy flows from the pump to the Stokes wave, and the
    anti-Stokes wave gives energy back. A sign slip anywhere in the kernel, the
    contraction or the causal direction flips both.
    """
    t = np.arange(SAMPLES) * DT
    pump_frequency = 2 * np.pi * 3.0e14
    shift = 2 * np.pi * 1.32e13  # about 440 cm^-1, inside the silica band
    window = np.exp(-(((t - 1000 * DT) / 500e-15) ** 2))
    pump = 1e9 * window * np.cos(pump_frequency * t)
    signs = []
    for side in (-1.0, +1.0):
        sideband = 1e6 * window * np.cos((pump_frequency + side * shift) * t)
        field = np.stack([pump + sideband, np.zeros(SAMPLES), np.zeros(SAMPLES)])
        total = response.evaluate(field, CHI3, kind="field", dt=DT)[0]
        alone = response.evaluate(
            np.stack([pump, np.zeros(SAMPLES), np.zeros(SAMPLES)]),
            CHI3,
            kind="field",
            dt=DT,
        )[0]
        driven = total - alone
        derivative = np.gradient(sideband, DT)
        signs.append(float(np.sum(driven * derivative)) * DT)
    stokes, anti_stokes = signs
    assert stokes > 0.0
    assert anti_stokes < 0.0
    # And the two are the same size to within the pump's own asymmetry.
    assert abs(stokes + anti_stokes) < 0.2 * abs(stokes)


def test_the_beat_free_drive_is_the_analytic_one(response):
    """`thg=False` drives the coherence with the cycle average, as the scalar does.

    The two field-grid drives of `raman_drive` differ only in the ``2 omega`` beat,
    so the slowly varying part of the response is the same and the difference is an
    oscillation at twice the carrier.
    """
    field = gaussian_field([1.0, 0.0, 0.0])
    with_beat = response.evaluate(field, CHI3, kind="field", dt=DT, thg=True)
    without = response.evaluate(field, CHI3, kind="field", dt=DT, thg=False)
    scale = float(np.max(np.abs(with_beat)))
    difference = with_beat - without
    assert 0.0 < float(np.max(np.abs(difference))) < scale
    # The beat integrates away over the window, which is what "cycle average" means.
    assert abs(float(np.sum(difference))) < 1e-3 * abs(
        float(np.sum(np.abs(difference)))
    )


# -------------------------------------------------------------------- the refusals


@pytest.mark.parametrize(
    "arguments,message",
    [
        ({"chi3": -1.0}, "non-negative chi3"),
        ({"chi3": float("nan")}, "non-negative chi3"),
        ({"dt": 0.0}, "sample spacing"),
    ],
)
def test_the_response_refuses_what_it_cannot_evaluate(response, arguments, message):
    """Every argument that is not one is named rather than approximated."""
    declared = {
        "field_t": gaussian_field([1.0, 0.0, 0.0]),
        "chi3": CHI3,
        "kind": "field",
        "dt": DT,
    }
    with pytest.raises((GridError, MaterialError), match=message):
        response.evaluate(**(declared | arguments))


def test_a_four_component_field_is_refused(response):
    """Three Cartesian components, and the refusal names the ceiling."""
    with pytest.raises(GridError, match="between one and 3 physical components"):
        response.evaluate(np.zeros((4, SAMPLES)), CHI3, kind="field", dt=DT)


def test_the_existing_scalar_silica_response_is_untouched(response, delays):
    """The scalar silica regressions are retained, as an assertion.

    The scalar silica response (optmat's Hollenbeck--Cantrell bundle shape since
    SEP 3) is a different fit reached by a different route, and the tensor model
    adds nothing to it and takes nothing from it.
    """
    scalar = response_for("SiO2")
    assert len(scalar.models) == 1
    assert scalar.traits.polarisation == "scalar"
    values = scalar.impulse(delays, 1.0)
    assert float(np.max(np.abs(values))) > 0.0
    # Not the tensor model's shape: two different fits of the same physics.
    tensor = response.shapes(delays)[0]
    normalised = values / float(np.max(np.abs(values)))
    assert not np.allclose(
        normalised, tensor / float(np.max(np.abs(tensor))), atol=1e-3
    )


def test_a_supplied_parameter_set_prints_its_time_constants_and_weights(response):
    # The optmat bundle prints its name and digest; a caller's Lin--Agrawal set has
    # neither, so it prints the three time constants, the weights and the integrals.
    assert repr(response) == "SilicaTensorResponse(f_R=0.245, 12.2 fs/32 fs/96 fs)"
    printed = str(response)
    assert "tau1, tau2:          12.2 fs, 32 fs" in printed
    assert "tau_b:               96 fs" in printed
    assert "0.75 h_a, 0.21 h_b, 0.04 h_a (anisotropic)" in printed
    assert "0.8333 trace, 0.1667 rank two" in printed

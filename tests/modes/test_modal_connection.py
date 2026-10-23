# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The solid-mode connection: the derivation, its certificate and its sign.

The derivation is that of ``docs/guide/waveguides/general-modal.md``, "The connection of
a basis
that moves along z". Every bound is stated before it is measured, and the two references
the derivation is checked against are independent of it: the **Bessel formula** of the
capillary, which the step-index connection must approach as the guide becomes the
Dirichlet set that formula describes, and the **direct projection of a frozen physical
field**, which fixes the sign without reference to any basis phase convention.
"""

import warnings
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import j1, jn_zeros

from modal_provider_cases import OMEGA, grin_provider, make_provider
from solprop.exceptions import GeneralModalBasisWarning, ModeError
from solprop.modes.connection import (
    CERTIFICATE_WARNING_RTOL,
    MAX_STENCIL_SPAN,
    STEP_FRACTION,
    ModalConnection,
    basis_connection,
    connection_nodes,
    modal_connection,
)
from solprop.modes.guide import StepIndexProvider, circular_pair, provider_channels
from solprop.modes.material import MaterialRegion
from solprop.modes.vector import ModalNodeRule, ModalProvider, ModalSample
from solprop.profiles import Linear, PiecewiseLinear

TAPER_LENGTH = 1e-3
"""Length of every declared taper here, m: a millimetre."""

MIDDLE = 0.5 * TAPER_LENGTH
"""Where every connection below is taken, m."""

CERTIFICATE_RTOL = 1e-8
"""Largest relative antisymmetry residual any case here is admitted at.

Dimensionless: ``antisymmetry / max|Gamma|``. The derivation makes the residual exactly
zero, so what is measured is the three-point stencil's own truncation and the providers'
solve noise. Measured 7.4e-11 on the step-index taper and 5.7e-9 on the graded-index one
at :data:`~solprop.modes.connection.STEP_FRACTION` (an independent re-measurement agreed
on both); this gate is 1.75x above the worst of them and fails on a broken quadrature
(which reads 1e-1), a lost gauge alignment (2.0) or a stencil at a fifth of its scale
(2.6e-2, which :func:`modal_connection` warns on above
:data:`~solprop.modes.connection.CERTIFICATE_WARNING_RTOL`), and not on a recompilation.
"""


def tapered_step_index(
    entrance=6e-6, ratio=1.35, indices=(1.45, 1.40), count=2, length=TAPER_LENGTH
):
    """Return a step-index provider whose core radius moves linearly along z."""
    descriptors = tuple(("HE", 1, m + 1, "even") for m in range(count))
    original = make_provider(radius=entrance, descriptors=descriptors, indices=indices)
    path = Linear(entrance, ratio * entrance, length)
    return StepIndexProvider(
        tuple(replace(mode, radius=path) for mode in original.modes),
        (
            MaterialRegion("core", 0.0, path, original.regions[0].material),
            MaterialRegion("cladding", path, None, original.regions[1].material),
        ),
    )


def tapered_grin(ratio=1.04, elements=32):
    """Return the parabolic graded-index sector with a moving core radius."""
    provider = grin_provider(elements=elements)
    radius = float(cast("float", provider.radius))
    return replace(provider, radius=Linear(radius, ratio * radius, TAPER_LENGTH))


GRIN_OMEGA = 2.0 * np.pi * 2.99792458e8 / 1.03e-6
"""Reference frequency of the graded-index sector, rad/s."""


@dataclass(frozen=True, eq=False)
class _Renormalised:
    """A provider whose channels lose their exact power normalisation along z.

    Every provider in this package normalises each channel to unit flux at every
    position, so its measured ``W'`` is zero and the flux self-terms with it. That
    is a property of these providers and **not** of the derivation: a tabulated or
    finite-element basis whose normalisation is only as good as its mesh, and a
    leaky channel whose flux is not conserved, both put the term back. This wrapper
    is the smallest thing that does: it scales channel ``n``'s fields by
    ``1 + rate_n (z - z0)``, which makes ``W_nn`` move at a rate this test knows in
    closed form.
    """

    inner: ModalProvider
    rates: tuple[float, ...]
    anchor: float

    def _scale(self, z: float):
        """Return the per-channel amplitude factor at one position."""
        return 1.0 + np.asarray(self.rates) * (float(z) - self.anchor)

    def node_rule(self, omega: float, z: float = 0.0, **kw) -> ModalNodeRule:
        """Delegate the rule unchanged."""
        return self.inner.node_rule(omega, z, **kw)

    def solve(self, omega: float, z: float, *, nodes: ModalNodeRule) -> ModalSample:
        """Return the inner solve with each channel's amplitude rescaled."""
        sample = self.inner.solve(omega, z, nodes=nodes)
        factor = self._scale(z)
        return replace(
            sample,
            electric=sample.electric * factor,
            magnetic=sample.magnetic * factor,
            flux_metric=sample.flux_metric * np.outer(factor, factor),
            backward_electric=None,
            backward_magnetic=None,
            reciprocal_gram=None,
        )

    @property
    def regions(self):
        """Delegate the regions unchanged."""
        return self.inner.regions

    @property
    def z_breakpoints(self):
        """Delegate the breakpoints unchanged."""
        return self.inner.z_breakpoints

    @property
    def omega_domain(self):
        """Delegate the band unchanged."""
        return self.inner.omega_domain

    @property
    def z_domain(self):
        """Delegate the z domain unchanged."""
        return self.inner.z_domain

    @property
    def identity(self):
        """Delegate the identity unchanged."""
        return self.inner.identity


def bessel_coefficient(count=2):
    """Return the capillary connection's dimensionless matrix (its Bessel formula)."""
    zeros = jn_zeros(0, count)
    signs = np.sign(j1(zeros))
    squared = zeros**2
    difference = squared[:, None] - squared[None, :]
    out = np.zeros((count, count))
    off = ~np.eye(count, dtype=bool)
    out[off] = 2.0 * np.outer(signs * zeros, signs * zeros)[off] / difference[off]
    return out


@pytest.fixture(scope="module")
def step_index_link():
    """The step-index taper's connection at the middle of its taper."""
    return modal_connection(tapered_step_index(), OMEGA, MIDDLE)


def test_the_connection_meets_the_metric_antisymmetry_its_derivation_predicts(
    step_index_link, record_property
):
    """``W Gamma + Gamma^H W = W'`` is the whole certificate, and it is measured.

    The bound is :data:`CERTIFICATE_RTOL`, declared before the run. ``W'`` is
    differenced from the stencil's own Gram matrices and never read off the pairing,
    so the residual is a statement about the sampling and not an identity of the
    arithmetic.
    """
    link = step_index_link
    scale = float(np.max(abs(link.matrix)))
    record_property("connection_scale_per_m", scale)
    record_property("connection_antisymmetry_relative", link.antisymmetry / scale)
    assert link.antisymmetry < CERTIFICATE_RTOL * scale
    # And for a provider that normalises exactly, the metric does not move at all,
    # which is why this basis's own self-terms are zero rather than merely small.
    assert float(np.linalg.norm(link.metric - np.eye(link.n_modes))) < 1e-12
    assert float(np.max(abs(link.metric_rate))) < 1e-8 * scale
    assert float(np.max(abs(link.self_terms))) < 1e-8 * scale


def test_the_connection_alone_conserves_the_flux_norm(step_index_link):
    """``d(c^H W c)/dz = 0`` under ``dc/dz = -Gamma c``, exactly.

    The consequence of the certificate above, and the reason the flux self-terms are
    kept: the rate is ``c^H (W' - W Gamma - Gamma^H W) c``, which the derivation
    makes zero. Zeroing the diagonal instead -- the capillary formula's own rule,
    correct for *its* basis -- would leave ``sum_n |c_n|^2 W'_nn``.
    """
    link = step_index_link
    rng = np.random.default_rng(20260922)
    amplitudes = rng.normal(size=link.n_modes) + 1j * rng.normal(size=link.n_modes)
    residual = (
        link.metric_rate
        - link.metric @ link.matrix
        - link.matrix.conj().T @ link.metric
    )
    rate = np.vdot(amplitudes, residual @ amplitudes)
    power = float(np.vdot(amplitudes, amplitudes).real)
    scale = float(np.max(abs(link.matrix))) * power
    assert abs(rate) < CERTIFICATE_RTOL * scale


def test_the_connection_is_real_and_antisymmetric_for_a_real_basis(step_index_link):
    """A real power-normalised basis makes the anti-Hermitian connection antisymmetric.

    Anti-Hermitian is the general statement; a basis the provider returns real makes
    it real, and then anti-Hermitian is antisymmetric with a zero diagonal -- which
    is the special case the capillary's Bessel formula writes down.
    """
    link = step_index_link
    scale = float(np.max(abs(link.matrix)))
    assert float(np.max(abs(np.imag(link.matrix)))) < 1e-12 * scale
    assert_allclose(link.matrix, -link.matrix.T, atol=1e-8 * scale, rtol=0)


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_every_entry_is_first_order_in_the_local_slope(factor):
    """Multiplying the taper's slope multiplies the connection, to the certificate.

    The connection is a derivative of the basis with respect to z, and the basis
    depends on z only through the radius, so every entry carries exactly one factor
    of ``a'`` and nothing else. The two tapers here run between the **same two
    radii** over lengths that differ by `factor`, and both are read where the radius
    is 1.175 times its entrance, so the only difference between the matrices is the
    slope. The bound is the certificate's own, not a fitted one.
    """
    reference = modal_connection(tapered_step_index(length=TAPER_LENGTH), OMEGA, MIDDLE)
    steeper = modal_connection(
        tapered_step_index(length=TAPER_LENGTH / factor), OMEGA, MIDDLE / factor
    )
    assert_allclose(steeper.matrix, factor * reference.matrix, rtol=1e-8, atol=0)


@pytest.mark.slow
def test_the_step_index_connection_approaches_the_capillary_bessel_formula():
    """The independent analytic reference: the Dirichlet limit, the Bessel formula.

    A step-index core's ``HE_1m`` fields become the Dirichlet Bessel set
    ``J_0(u_m r/a)`` as ``V`` grows, and that set is exactly the one the capillary's
    connection was derived on, so this connection -- computed from nothing but
    reciprocity and the sampled power normalisation -- must approach the Bessel
    formula. It does, and the departure **halves as V doubles**, which is the
    Goos-Hanchen penetration depth's own ``1/V``.

    The gate is declared first: the departure at ``V = 55.7`` must be below 2 % and
    each doubling must reduce it by at least 1.7. At a *finite* ``V`` the two are
    genuinely different -- 4.6 % apart at ``V = 20`` -- which is why the formula is
    not transplanted.
    """
    expected = abs(bessel_coefficient()[0, 1])
    measured = []
    for entrance, indices in (
        (6.0e-6, (1.45, 1.00)),
        (1.2e-5, (1.45, 1.00)),
        (2.4e-5, (1.45, 1.00)),
    ):
        provider = tapered_step_index(entrance=entrance, indices=indices)
        link = modal_connection(provider, OMEGA, MIDDLE)
        radius = float(provider.modes[0]._radius(MIDDLE))
        slope = 0.35 * entrance / TAPER_LENGTH
        measured.append(abs(float(np.real(link.matrix[0, 1]))) / (slope / radius))
    departures = [abs(one / expected - 1.0) for one in measured]
    assert departures[0] < 2e-2
    assert departures[0] / departures[1] > 1.7
    assert departures[1] / departures[2] > 1.7


@pytest.mark.slow
def test_the_connection_transports_a_frozen_field_and_fixes_its_sign():
    """The sign is established against a direct projection, not asserted.

    A physical field frozen at the entrance is propagated on the connection alone
    with ``dc/dz = -Gamma c`` and compared with the coefficients the **same** field
    has on the basis at the exit, taken by direct projection. The two agree to second
    order in the total contraction -- the retained set's own truncation, as for the
    capillary -- and propagating with ``+Gamma`` instead is two orders worse, which is
    what fixes the sign.

    Gates declared first: the departure falls by at least 3.5 per doubling of the
    contraction, and the wrong sign is at least 50 times the right one.
    """
    provider = tapered_step_index()
    entrance = 1e-5
    departures = []
    for delta in (1e-5, 2e-5, 4e-5):
        propagated, projected = _transport(provider, entrance, delta, sign=-1.0)
        departures.append(float(np.linalg.norm(propagated - projected)))
    assert departures[1] / departures[0] > 3.5
    assert departures[2] / departures[1] > 3.5
    wrong, projected = _transport(provider, entrance, 4e-5, sign=+1.0)
    assert float(np.linalg.norm(wrong - projected)) > 50.0 * departures[-1]


def _transport(provider, entrance, delta, *, sign, steps=8):
    """Propagate a frozen field's coefficients and project the same field directly."""
    positions = np.linspace(entrance, entrance + delta, steps + 1)
    rules = connection_nodes(provider, OMEGA, positions)
    first = provider.solve(OMEGA, float(positions[0]), nodes=rules[0])
    amplitudes = np.zeros(first.electric.shape[2], dtype=complex)
    amplitudes[0] = 1.0
    field = (first.electric @ amplitudes, first.magnetic @ amplitudes)
    width = float(positions[1] - positions[0])
    cache: dict[float, np.ndarray] = {}

    def slope(at):
        key = round(float(at), 13)
        if key not in cache:
            cache[key] = sign * modal_connection(provider, OMEGA, key).matrix
        return cache[key]

    for start in positions[:-1]:
        first_stage = slope(start) @ amplitudes
        middle = slope(start + width / 2)
        second = middle @ (amplitudes + width * first_stage / 2)
        third = middle @ (amplitudes + width * second / 2)
        fourth = slope(start + width) @ (amplitudes + width * third)
        amplitudes = (
            amplitudes + width * (first_stage + 2 * second + 2 * third + fourth) / 6
        )
    final = provider.solve(OMEGA, float(positions[-1]), nodes=rules[-1])
    weights = rules[-1].weights
    overlap = (
        np.einsum("qm,q,q->m", final.magnetic[:, 1].conj(), field[0][:, 0], weights)
        - np.einsum("qm,q,q->m", final.magnetic[:, 0].conj(), field[0][:, 1], weights)
        + np.einsum("qm,q,q->m", final.electric[:, 0].conj(), field[1][:, 1], weights)
        - np.einsum("qm,q,q->m", final.electric[:, 1].conj(), field[1][:, 0], weights)
    ) / 4
    return amplitudes, np.linalg.solve(final.flux_metric, overlap)


def test_a_provider_whose_normalisation_moves_carries_a_real_flux_self_term():
    """The diagonal is not zero in general, and this is what puts it back.

    Every provider in this package normalises exactly, so its ``W'`` is zero and its
    self-terms with it. Scaling channel ``n`` by ``1 + rate_n (z - z0)`` makes
    ``W_nn`` move at a rate known in closed form, and the derivation says the
    connection's diagonal is then ``W'_nn / (2 W_nn)`` -- the flux self-term. It is
    measured here against that closed form, and the flux identity still holds
    exactly; **zeroing** the diagonal instead breaks it by the self-terms, which is
    the term the connection must not omit.
    """
    rates = (7.0e3, -3.0e3)
    provider = _Renormalised(tapered_step_index(), rates, MIDDLE)
    link = modal_connection(provider, OMEGA, MIDDLE)
    # At the anchor the scale factors are one, so W is the identity and the closed
    # form for the diagonal is the declared rate itself.
    assert_allclose(np.real(link.self_terms), rates, rtol=1e-6, atol=0)
    assert_allclose(
        np.real(np.diag(link.metric_rate)), 2.0 * np.array(rates), rtol=1e-6
    )
    scale = float(np.max(abs(link.matrix)))
    assert link.antisymmetry < CERTIFICATE_RTOL * scale
    # And the identity fails by exactly the self-terms when they are dropped.
    dropped = link.matrix - np.diag(link.self_terms)
    residual = link.metric @ dropped + dropped.conj().T @ link.metric - link.metric_rate
    assert_allclose(np.real(np.diag(residual)), -2.0 * np.array(rates), rtol=1e-6)


def test_a_declared_rotating_basis_is_a_relabelling_and_converts_nothing():
    """The pure change-of-basis test of the connection, for ``c = T(z) d``.

    With ``-T^{-1}T'`` carried beside the transformed generator a z-dependent
    unitary change of coordinates moves no power between physical channels. Both
    sides are integrated by the same fourth-order scheme on the same schedule, so
    what is compared is the equation and not the integrator.

    The bounds are declared first: the rotated propagation must reproduce the direct
    one to 1e-10 of the state -- the scheme's own accumulated error is 2.5e-13 here
    -- and **dropping** the term must move it by a fifth of its own size or more.
    """
    generator = np.diag([3.0e3j, 2.0e3j])
    rate, length, steps = 1.7e4, 3.0e-4, 4096
    width = length / steps

    def turn(z):
        angle = rate * z
        return np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=complex,
        )

    def slope(z):
        angle = rate * z
        return rate * np.array(
            [[-np.sin(angle), -np.cos(angle)], [np.cos(angle), -np.sin(angle)]],
            dtype=complex,
        )

    def rotated_generator(z, *, connected):
        matrix = turn(z)
        turned = matrix.conj().T @ generator @ matrix
        return turned - basis_connection(matrix, slope(z)) if connected else turned

    def advance(state, at, operator):
        """One classical fourth-order step of ``dc/dz = A(z) c``."""
        first = operator(at) @ state
        middle = operator(at + width / 2)
        second = middle @ (state + width * first / 2)
        third = middle @ (state + width * second / 2)
        fourth = operator(at + width) @ (state + width * third)
        return state + width * (first + 2 * second + 2 * third + fourth) / 6

    direct = np.array([1.0 + 0.0j, 0.0])
    connected = turn(0.0).conj().T @ direct
    dropped = connected.copy()
    for index in range(steps):
        here = index * width
        direct = advance(direct, here, lambda _: generator)
        connected = advance(
            connected, here, lambda at: rotated_generator(at, connected=True)
        )
        dropped = advance(
            dropped, here, lambda at: rotated_generator(at, connected=False)
        )
    assert_allclose(turn(length) @ connected, direct, atol=1e-10, rtol=0)
    assert float(np.linalg.norm(turn(length) @ dropped - direct)) > 0.2


def test_a_declared_rotating_basis_connection_is_anti_hermitian():
    """``T^H T'`` is anti-Hermitian for any unitary path, with an imaginary diagonal.

    The general statement of the module's derivation: differentiating ``T^H T = I``
    gives ``T^H T' + (T^H T')^H = 0``, so a declared moving basis contributes only a
    gauge phase rate on the diagonal and no amplitude change at all.
    """
    rate = 9.0e3
    phase = np.diag([1.0 + 0.0j, np.exp(0.4j)])
    slope = np.diag([0.0 + 0.0j, 1j * rate * np.exp(0.4j)])
    link = basis_connection(phase, slope)
    assert_allclose(link, -link.conj().T, atol=1e-12, rtol=0)
    assert_allclose(np.real(np.diag(link)), np.zeros(2), atol=1e-12, rtol=0)
    assert_allclose(np.imag(np.diag(link)), [0.0, rate], rtol=1e-12)


def test_a_frozen_node_rule_misreads_a_moving_interface_by_a_whole_self_term():
    """Why the panels are split at every stencil position's own edges.

    A rule built at one position and reused at the others leaves its panel edge off
    the moving material interface, and the Poynting integrand's jump there is then
    integrated by a Gauss rule that does not see it. The error is **first order in
    the displacement**, so it is first order in the step and survives the difference
    undiminished: the certificate this module reports rises by orders and the metric
    reads a rate that is not there.
    """
    provider = tapered_step_index()
    split = modal_connection(provider, OMEGA, MIDDLE)
    frozen = provider.node_rule(OMEGA, MIDDLE)
    positions = (MIDDLE - split.step, MIDDLE, MIDDLE + split.step)
    samples = [provider.solve(OMEGA, one, nodes=frozen) for one in positions]
    rate = (samples[2].flux_metric - samples[0].flux_metric) / (2.0 * split.step)
    # The split rule measures a metric that does not move; the frozen one measures a
    # rate of order the connection itself, which is a self-term that is not there.
    assert float(np.max(abs(split.metric_rate))) < 1e-8 * float(
        np.max(abs(split.matrix))
    )
    assert float(np.max(abs(rate))) > 1e-3 * float(np.max(abs(split.matrix)))


def test_the_graded_index_sector_certifies_on_its_own_element_mesh(record_property):
    """A finite-element provider's panels move with its core, and the rule follows.

    The graded-index sector is a P1 field whose slope breaks at every element edge
    and whose whole mesh scales with the radius, so the union of the stencil's
    declared edges -- not of its material radii -- is what makes each quadrature
    exact. Splitting at the interface alone reads a flux metric 0.39 from the
    identity; this measures what the element mesh gives instead.
    """
    link = modal_connection(tapered_grin(), GRIN_OMEGA, MIDDLE)
    scale = float(np.max(abs(link.matrix)))
    record_property("grin_connection_scale_per_m", scale)
    record_property("grin_connection_antisymmetry_relative", link.antisymmetry / scale)
    assert float(np.linalg.norm(link.metric - np.eye(link.n_modes))) < 1e-12
    assert link.antisymmetry < CERTIFICATE_RTOL * scale
    # The raw matrix is anti-Hermitian only to the variational solve's own noise,
    # which is an order above the certificate: the certificate is a statement about
    # one stencil's three Gram matrices, which share their discretisation error,
    # while this compares two entries of one matrix that do not.
    assert_allclose(link.matrix, -link.matrix.conj().T, atol=1e-7 * scale, rtol=0)


def test_the_connection_is_taken_in_the_declared_channel_basis():
    """A declared unitary basis reaches the connection as it reaches the tables.

    The transform is applied after the tracking gauge, in the same fixed order
    :func:`solprop.translate.modal_tables.composed_gauge` documents, so the
    connection is ``U^H Gamma U`` -- a similarity, because a *constant* declaration
    adds no term of its own.
    """
    provider = tapered_step_index(indices=(1.45, 1.40), count=1)
    pair = make_provider(
        radius=6e-6,
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        indices=(1.45, 1.40),
    )
    path = Linear(6e-6, 1.35 * 6e-6, TAPER_LENGTH)
    moving = StepIndexProvider(
        tuple(replace(mode, radius=path) for mode in pair.modes),
        (
            MaterialRegion("core", 0.0, path, pair.regions[0].material),
            MaterialRegion("cladding", path, None, pair.regions[1].material),
        ),
    )
    basis = circular_pair(provider_channels(moving), (0, 1))
    plain = modal_connection(moving, OMEGA, MIDDLE, degenerate_blocks=((0, 1),))
    turned = modal_connection(
        moving, OMEGA, MIDDLE, degenerate_blocks=((0, 1),), basis=basis
    )
    matrix = basis.transform
    assert_allclose(
        turned.matrix,
        matrix.conj().T @ plain.matrix @ matrix,
        atol=1e-9 * max(float(np.max(abs(plain.matrix))), 1.0),
        rtol=0,
    )
    assert provider.z_domain == moving.z_domain


def test_the_stencil_never_crosses_a_declared_kink():
    """A derivative across a corner is the derivative of neither side.

    The z scale a step is taken from is the interval between the declared
    breakpoints the position lies in, so a stencil beside a kink shifts onto its own
    side rather than straddling it.
    """
    entrance = 6e-6
    path = PiecewiseLinear(
        np.array([0.0, 0.4 * TAPER_LENGTH, TAPER_LENGTH]),
        np.array([entrance, 1.3 * entrance, entrance]),
    )
    original = make_provider(
        radius=entrance,
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 2, "even")),
        indices=(1.45, 1.40),
    )
    provider = StepIndexProvider(
        tuple(replace(mode, radius=path) for mode in original.modes),
        (
            MaterialRegion("core", 0.0, path, original.regions[0].material),
            MaterialRegion("cladding", path, None, original.regions[1].material),
        ),
    )
    kink = 0.4 * TAPER_LENGTH
    below = modal_connection(provider, OMEGA, kink - 1e-9)
    above = modal_connection(provider, OMEGA, kink + 1e-9)
    # The two one-sided slopes differ in sign, which a straddling stencil averages
    # away; both stencils lie wholly inside their own segment.
    assert below.step == pytest.approx(STEP_FRACTION * 0.4 * TAPER_LENGTH)
    assert above.step == pytest.approx(STEP_FRACTION * 0.6 * TAPER_LENGTH)
    assert float(np.real(below.matrix[0, 1])) * float(np.real(above.matrix[0, 1])) < 0.0


@pytest.mark.parametrize("step", [0.0, -1e-9, float("inf")])
def test_a_step_outside_its_range_is_refused_by_name(step):
    """Zero, negative and infinite steps are named, not silently clamped."""
    with pytest.raises(ModeError, match="positive finite step"):
        modal_connection(tapered_step_index(), OMEGA, MIDDLE, step=step)


def test_a_coarse_declared_step_warns_against_its_own_certificate():
    """A step at a fifth of the scale is admitted and warns by name.

    At ``h/L = 0.2`` the certificate reads 2.6e-2 with the connection itself 1.2e-2
    wrong, still inside ``MAX_STENCIL_SPAN``; the certificate is what says so, and the
    default step does not warn.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeneralModalBasisWarning)
        fine = modal_connection(tapered_step_index(), OMEGA, MIDDLE)
    assert fine.antisymmetry <= CERTIFICATE_WARNING_RTOL * np.max(abs(fine.matrix))
    with pytest.warns(GeneralModalBasisWarning, match="too coarse for this provider"):
        coarse = modal_connection(
            tapered_step_index(),
            OMEGA,
            MIDDLE,
            step=0.8 * MAX_STENCIL_SPAN * TAPER_LENGTH,
        )
    assert coarse.antisymmetry > CERTIFICATE_WARNING_RTOL * np.max(abs(coarse.matrix))


def test_a_stencil_wider_than_its_scale_is_refused_by_name():
    """A secant across a quarter of the scale is not a derivative."""
    with pytest.raises(ModeError, match="is a secant"):
        modal_connection(
            tapered_step_index(),
            OMEGA,
            MIDDLE,
            step=2.0 * MAX_STENCIL_SPAN * TAPER_LENGTH,
        )


def test_a_rule_that_declares_no_panel_edges_is_refused_by_name():
    """The refinement needs the rule's own edges, resolution and truncation radius."""
    provider = tapered_step_index()
    rule = provider.node_rule(OMEGA, MIDDLE)
    bare = ModalNodeRule(rule.xy, rule.weights, rule.region_index, {"omega": OMEGA})
    with pytest.raises(ModeError, match="radial panel rule"):
        connection_nodes(provider, OMEGA, (MIDDLE,), nodes=bare)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("matrix", np.zeros((2, 3), dtype=complex)),
        ("metric", np.zeros((3, 3), dtype=complex)),
        ("metric_rate", np.full((2, 2), np.nan, dtype=complex)),
    ],
)
def test_a_connection_record_checks_its_own_matrices(name, value):
    """Three finite square matrices of one width, or a refusal naming the field."""
    fields = {
        "matrix": np.eye(2, dtype=complex),
        "metric": np.eye(2, dtype=complex),
        "metric_rate": np.zeros((2, 2), dtype=complex),
    }
    fields[name] = value
    with pytest.raises(ModeError, match=f"connection's {name}"):
        ModalConnection(**fields, antisymmetry=0.0, step=1e-9, z=0.0, omega=1e15)


@pytest.mark.parametrize(
    ("name", "value"),
    [("antisymmetry", -1.0), ("step", 0.0), ("omega", -1.0), ("z", float("nan"))],
)
def test_a_connection_record_checks_its_own_scalars(name, value):
    """Every scalar is finite and in range, or the field is named."""
    fields = {
        "matrix": np.eye(2, dtype=complex),
        "metric": np.eye(2, dtype=complex),
        "metric_rate": np.zeros((2, 2), dtype=complex),
        "antisymmetry": 0.0,
        "step": 1e-9,
        "z": 0.0,
        "omega": 1e15,
    }
    fields[name] = value
    with pytest.raises(ModeError, match=name):
        ModalConnection(**fields)


def test_a_declared_basis_connection_checks_its_two_matrices():
    """``T`` and ``T'`` are square and equally wide, or the call is refused."""
    with pytest.raises(ModeError, match="square transform"):
        basis_connection(np.eye(2, dtype=complex), np.zeros((3, 3), dtype=complex))


def test_the_connection_record_prints_without_an_array(step_index_link):
    """The one-line form and the block both describe, never dump."""
    assert "ModalConnection" in repr(step_index_link)
    assert "1/m" in repr(step_index_link)
    block = str(step_index_link)
    assert "antisymmetry" in block
    assert "self terms" in block


def test_a_specified_mode_declares_no_field_to_pair_and_so_has_no_connection():
    """How the specified-mode route relates to the connection.

    A :class:`~solprop.nonlinear.gnlse.SpecifiedMode` built from a moving guide
    carries a **dispersion path** and an **area path** and no transverse field at
    all: its ``field`` is a surrogate disc of radius ``sqrt(A_eff(z)/pi)``, which
    exists for the nonlinear coefficient and is not the mode. So
    ``<psi_m | d_z psi_n>`` cannot be formed from one, and
    :func:`~solprop.modes.connection.modal_connection` takes a **provider**, which a
    specified fibre is not.

    A *single* moving specified channel needs no connection: a one-by-one
    anti-Hermitian matrix is a pure gauge phase and the declaration owns its own
    gauge. A **multi-channel** moving set does need one, and this pins what happens
    today -- it propagates on the diagonal with the term silently absent, which is
    the mode-set route's standing behaviour for every moving multimode guide. This is a
    known limitation, pinned here rather than changed.
    """
    from solprop.modes import MarcatiliMode  # noqa: PLC0415
    from solprop.nonlinear.gnlse import SpecifiedMode  # noqa: PLC0415

    taper = Linear(50e-6, 40e-6, 2e-4)
    modes = [MarcatiliMode(taper, "He", 1.0, m=index) for index in (1, 2)]
    fibres = [SpecifiedMode.from_mode(mode) for mode in modes]
    # Both move, both are scale invariant, and neither offers a provider: the
    # mode-set route carries them through `taper_scale` alone.
    assert all(not fibre.z_constant for fibre in fibres)
    assert all(fibre.scale_invariant for fibre in fibres)
    # It carries none of the provider surface the connection is measured from, so
    # it is refused by the signature rather than by a message: there is nothing to
    # sample, no node rule to refine and no field to pair.
    missing = [
        name
        for name in ("solve", "node_rule", "regions", "z_domain", "z_breakpoints")
        if not hasattr(fibres[0], name)
    ]
    assert missing == ["solve", "node_rule", "regions", "z_domain", "z_breakpoints"]
    assert callable(modal_connection)
    # The connection this pair *would* need is the capillary's own, and it is not
    # small: `wall_connection` reads it directly from the profile the fibres were
    # built from.
    from solprop.modes import wall_connection  # noqa: PLC0415

    link = wall_connection(modes, 1e-4)
    assert float(np.max(abs(link))) > 1e3


@dataclass(frozen=True, eq=False)
class _Edgeless:
    """A provider whose node rule declares no panel edges inside its own domain.

    The refinement is built from the union of every stencil position's **declared**
    edges, so a rule that declares none leaves it nothing to split at. No provider
    in this package returns such a rule; this is the smallest thing that does.
    """

    inner: ModalProvider

    def node_rule(self, omega: float, z: float = 0.0, **kw) -> ModalNodeRule:
        """Return the inner rule with its declared edges emptied."""
        rule = self.inner.node_rule(omega, z, **kw)
        return ModalNodeRule(
            rule.xy,
            rule.weights,
            rule.region_index,
            {**rule.identity, "interfaces": []},
        )

    def solve(self, omega: float, z: float, *, nodes: ModalNodeRule) -> ModalSample:
        """Delegate the solve unchanged."""
        return self.inner.solve(omega, z, nodes=nodes)

    @property
    def regions(self):
        """Delegate the regions unchanged."""
        return self.inner.regions

    @property
    def z_breakpoints(self):
        """Delegate the breakpoints unchanged."""
        return self.inner.z_breakpoints

    @property
    def omega_domain(self):
        """Delegate the band unchanged."""
        return self.inner.omega_domain

    @property
    def z_domain(self):
        """Delegate the z domain unchanged."""
        return self.inner.z_domain

    @property
    def identity(self):
        """Delegate the identity unchanged."""
        return self.inner.identity


def test_a_rule_with_no_edges_inside_its_domain_is_refused_by_name():
    """The refinement splits at declared edges, and needs at least one."""
    with pytest.raises(ModeError, match="declares none inside"):
        connection_nodes(_Edgeless(tapered_step_index()), OMEGA, (MIDDLE,))

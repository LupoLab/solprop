# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The condensed Robin wall operator of an HE_1m block.

Every numerical bound in this module is stated before the measurement it gates, in the
docstring of the test that takes it, and every one of them is an *analytic* prediction
of the deviation rather than a tolerance chosen after a run.

The independent references are: complex roots of the exact outgoing radial equation
``q J_1(q) + i eta J_0(q) = 0``, solved by a root finder that knows nothing about the
condensation; a one-sided difference of the basis function at the wall; this package's
own :class:`~solprop.modes.capillary.MarcatiliMode` effective index; and the committed
wall-stiffness rows of an independent prototype.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad, solve_ivp
from scipy.linalg import sqrtm
from scipy.optimize import newton, root
from scipy.special import jn_zeros, jv

import solprop.modes.capillary as capillary_module
from solprop.constants import EPS0, C
from solprop.exceptions import GrazingIncidenceWarning, ModeError
from solprop.modes import (
    CoherentWall,
    MarcatiliMode,
    ModeCollection,
    RectMode,
    TubeLattice,
    VincettiMode,
    WallLimits,
    capillary,
    vector_wall_operator,
    wall_connection,
    wall_operator,
    wall_square_root_metric,
)
from solprop.modes.capillary import ModeKind, _cladding_term_and_derivative
from solprop.profiles import FromCallable, Linear, ZProfile

ORACLE = Path(__file__).resolve().parents[1] / "data" / "wall_oracle"

STIFFNESS = ORACLE / "stiffness.json"
"""Rows of an independent prototype's wall-stiffness study: sixty wall blocks."""

VECTOR_ROOTS = ORACLE / "vector_roots.json"
"""Exact outgoing vector boundary roots of a 150 micrometre bore for TE, TM, HE and EH,
and the mixed-family angular Gram, computed by an independent prototype."""


def block(count, radius=125e-6, gas: str | None = "He", pressure=1.0, **kwargs):
    """Return a contiguous HE_1m Marcatili block of `count` channels."""
    return [
        MarcatiliMode(radius, gas, pressure, m=index + 1, **kwargs)
        for index in range(count)
    ]


def robin_root(guess, eta):
    """Solve q J_1(q) + i eta J_0(q) = 0 near `guess`, independently of the block."""

    def residual(value):
        q = np.array([complex(value[0], value[1])])
        out = complex((q * jv(1, q) + 1j * eta * jv(0, q))[0])
        return [out.real, out.imag]

    found = root(residual, [guess.real, guess.imag])
    assert found.success
    return complex(found.x[0], found.x[1])


def robin_root_of_order(order, index, eta):
    """Solve `q J_{p+1}(q) = (p - i eta) J_p(q)` near the `index`-th zero of `J_p`.

    A complex Newton iteration rather than the two-real-component solve above: the
    general-order equation is analytic in `q`, and a real solver on its real and
    imaginary parts stalls at these radii.
    """

    def residual(q):
        return q * jv(order + 1, q) - (order - 1j * eta) * jv(order, q)

    start = complex(jn_zeros(order, index)[-1]) * (1.0 - 1j / eta)
    return complex(newton(residual, start, tol=1e-14, maxiter=200))


# ------------------------------------------------------------------ the operator


def test_the_trace_vector_is_the_basis_derivative_at_the_wall():
    """`h_m` is `phi_m'(1)`, to the 1e-4 of a one-sided difference of step 1e-6.

    The basis is `phi_m(x) = sqrt(2) J_0(u_m x)/|J_1(u_m)|`, so its derivative at the
    wall is `-sqrt(2) sgn(J_1(u_m)) u_m`: the **sign** is the whole content of the
    claim, and a difference resolves it to first order in the step, which is 1e-6
    relative on a quantity of order ten.
    """
    operator = wall_operator(block(6), np.array([2.356e15]))
    zeros = jn_zeros(0, 6)
    step = 1e-6
    measured = [
        (
            np.sqrt(2.0) * jv(0, zero) / abs(jv(1, zero))
            - np.sqrt(2.0) * jv(0, zero * (1.0 - step)) / abs(jv(1, zero))
        )
        / step
        for zero in zeros
    ]
    assert np.allclose(operator.trace_vector, np.array(measured), rtol=1e-4)


@pytest.mark.parametrize("count", [10, 20, 40])
def test_the_condensed_eigenvalues_converge_to_independent_robin_roots(count):
    """The four lowest transverse eigenvalues approach the exact roots as 1/N.

    Bound, stated first: the condensation's quasi-static error falls like 1/N, so the
    relative departure of each of the four lowest `q` from the independently solved
    root of `q J_1(q) + i eta J_0(q) = 0` is below `1e-4/N` and at least halves when
    `N` doubles. Nothing here is a Maxwell-accuracy claim: it says the condensed
    operator is the truncation of the exact root equation it was derived from.
    """
    radius, wavelength, cladding = 150e-6, 800e-9, 1.3
    omega = np.array([2.0 * np.pi * C / wavelength])
    modes = block(count, radius, None, 0.0, cladding=cladding)
    operator = wall_operator(modes, omega, validity_limit=10.0)
    carrier = complex(operator.wavenumber[0])
    matrix = operator.generator[:, :, 0] * (2.0 * carrier * radius**2) / 1j
    eigenvalues = np.sort_complex(-np.linalg.eigvals(matrix))
    eta = complex(radius * operator.impedance[0])
    exact = np.array([robin_root(complex(zero), eta) for zero in jn_zeros(0, 4)])
    departure = np.abs(np.sqrt(eigenvalues[:4]) - exact) / np.abs(exact)
    assert np.all(departure < 1e-4 / count), departure


def test_the_single_mode_block_reproduces_the_capillary_loss_to_its_derived_bound():
    """With N = 1 the block's HE_11 decay is this package's Marcatili loss.

    Bound, stated first and **derived**, not fitted. Writing `t = v_n/(k_0 a)` and
    `eta = 1/t`, the condensed diagonal is
    `-i u_m^2 (1 - 2/(2N - i eta))/(2 k a^2)`, whose imaginary part is
    `-2 u_m^2 t/(1 + 4 N^2 t^2)` against Marcatili's `-2 u_m^2 t`; and the paraxial
    reduction divides by `2k` where the exact square root divides by `2 Re beta`,
    which is a further `-u_m^2/(2(ka)^2)`. So the relative deviation of the block's
    field-loss rate from `(omega/c) Im n_eff` is
    `-(4 N^2 t^2 + u_m^2/(2 (ka)^2))` to leading order. The first term is the square
    of the `2N/|a k_eff|` quasi-static ratio the block records as a validity limit,
    which is why raising N cannot by itself improve the condensation. This asserts
    the measurement agrees with that prediction to 1 %.
    """
    for radius, wavelength in ((125e-6, 800e-9), (25e-6, 1e-6), (150e-6, 3e-6)):
        omega = np.array([2.0 * np.pi * C / wavelength])
        mode = MarcatiliMode(radius, None, 0.0, m=1)
        operator = wall_operator([mode], omega, validity_limit=math.inf)
        measured = -float(np.real(operator.generator[0, 0, 0]))
        marcatili = float(omega[0] / C * np.imag(mode.neff(omega)[0]))
        inverse_eta = 1.0 / abs(radius * complex(operator.impedance[0]))
        carrier = abs(complex(operator.wavenumber[0])) * radius
        predicted = -(4.0 * inverse_eta**2 + mode.u_nm**2 / (2.0 * carrier**2))
        assert measured / marcatili - 1.0 == pytest.approx(predicted, rel=1e-2)


def test_every_channel_of_a_block_carries_its_own_marcatili_loss():
    """Each diagonal entry is that channel's loss, to the same derived bound.

    Bound, stated first: the same `-(4 N^2 t^2 + u_m^2/(2 (ka)^2))` as the
    single-mode case, now with the block's own `N` and the channel's own Bessel zero,
    predicted to 5 %. This is the weakly coupled limit of the propagation test in
    ``tests/driver/test_coherent_capillary.py``: the off-diagonal moves power
    between channels, and the diagonal is what each channel loses to the wall.
    """
    radius, omega = 125e-6, np.array([2.356e15])
    modes = block(4, radius, None, 0.0)
    operator = wall_operator(modes, omega, validity_limit=math.inf)
    inverse_eta = 1.0 / abs(radius * complex(operator.impedance[0]))
    carrier = abs(complex(operator.wavenumber[0])) * radius
    for index, mode in enumerate(modes):
        measured = -float(np.real(operator.generator[index, index, 0]))
        marcatili = float(omega[0] / C * np.imag(mode.neff(omega)[0]))
        predicted = -(
            4.0 * len(modes) ** 2 * inverse_eta**2 + mode.u_nm**2 / (2.0 * carrier**2)
        )
        assert measured / marcatili - 1.0 == pytest.approx(predicted, rel=5e-2)


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore:The maximum number of subdivisions")
def test_the_block_reproduces_the_prototype_stiffness_rows():
    """Every row of the prototype's stiffness study comes out of the production block.

    Bound, stated first: the production operator and the prototype are the
    same arithmetic in a different order, so the three spectral quantities the
    receipt records -- the rank-one wall term's spectral radius, that of its
    off-diagonal remainder, and the highest channel's Marcatili loss -- agree to
    1e-12 relative, which is the rounding of an eigenvalue solve of a rank-one
    perturbation at these magnitudes. All sixty rows are checked, five wavelengths
    from 0.8 to 12 micrometres, both cladding indices and N from 8 to 160.
    """
    rows = json.loads(STIFFNESS.read_text())
    assert len(rows) == 60
    for row in rows:
        omega = np.array([2.0 * np.pi * C / (row["wavelength_um"] * 1e-6)])
        modes = block(
            row["N"], row["radius_m"], None, 0.0, cladding=row["cladding_index"]
        )
        operator = wall_operator(modes, omega, validity_limit=100.0)
        rank_one = operator.coupling[0] * np.outer(
            operator.trace_vector, operator.trace_vector
        )
        whole = float(max(abs(np.linalg.eigvals(rank_one))))
        off = rank_one - np.diag(np.diag(rank_one))
        remainder = float(max(abs(np.linalg.eigvals(off))))
        highest = (
            modes[-1].u_nm ** 2
            * float(np.real(omega[0] / C / operator.impedance[0]))
            / (float(np.real(operator.wavenumber[0])) ** 2 * row["radius_m"] ** 3)
        )
        assert whole == pytest.approx(row["whole_wall"]["rho_per_m"], rel=1e-12)
        assert remainder == pytest.approx(row["offdiagonal"]["rho_per_m"], rel=1e-12)
        assert highest == pytest.approx(row["alpha_N_marcatili_per_m"], rel=1e-12)


def test_the_block_is_passive_and_its_hermitian_part_is_rank_one():
    """The Hermitian part has one negative eigenvalue and N - 1 zeros, to 1e-12.

    Bound, stated first: `(G + G^H)/2 = -Re(eta) h h^T/(|k| a^2 (4N^2 + |eta|^2))`
    exactly, which is negative semidefinite of rank one, so the largest eigenvalue is
    zero to the 1e-12 of the block's own scale and the trace equals the one negative
    eigenvalue. That the sign is negative is passivity: the wall absorbs.
    """
    operator = wall_operator(block(5), np.array([2.356e15]))
    matrix = operator.generator[:, :, 0]
    hermitian = 0.5 * (matrix + np.conj(matrix.T))
    eigenvalues = np.linalg.eigvalsh(hermitian)
    scale = float(np.max(abs(matrix)))
    assert eigenvalues[-1] == pytest.approx(0.0, abs=1e-12 * scale)
    assert eigenvalues[0] < 0.0
    assert float(np.trace(hermitian).real) == pytest.approx(eigenvalues[0], rel=1e-10)
    assert np.allclose(eigenvalues[1:], 0.0, atol=1e-12 * scale)


def test_the_square_root_flux_identity_holds_on_its_stated_domain():
    """`d(c^H W c)/dz` is `-c^H (M - M^H) c/(2ik)`, to a bound derived from rounding.

    Bound, stated first. With the **computed** root `X` and `W = (X + X^H)/(2k)`,
    `W X - X^H W = (X^2 - X^H^2)/(2k)`, so the left side is exactly
    `i c^H (X^2 - X^H^2) c/(2k)` and the right one `i c^H (M - M^H) c/(2k)`: they
    differ by `i c^H (R - R^H) c/(2k)` with `R = X^2 - M`, the square root's own
    residual, which the test forms (to `gamma(n+1)` of `|X|^2 + |M|`, entrywise). The
    rest is the evaluation of the two sides: flow, metric and quadratic forms carry
    at most `3n + 4` roundings a term, each complex product two more (Higham,
    *Accuracy and Stability*, lemma 3.5), so `gamma(4n + 8)` of
    `|c|^2 (2 ||X|| ||W|| + ||M||/k)` in absolute-value Frobenius norms. The exact left
    side is real, so its imaginary part is that rounding alone. The two sides are
    a difference of terms of size `k |c|^2`, about eight orders above the ~2 they
    cancel to, so a relative tolerance fixed near `u k |c|^2 / 2` holds on one
    platform's last bits and not another's (an x86-64 server missed 1e-9 by two).
    The **domain** is the one the docstring of
    :func:`wall_square_root_metric` states: this says nothing about positivity of `W`,
    about a complex carrier, or about a varying section.
    """
    generator = np.random.default_rng(0)
    operator = wall_operator(block(5), np.array([2.356e15]))
    matrix = operator.generator[:, :, 0]
    carrier = float(np.real(operator.wavenumber[0]))
    root_matrix, metric = wall_square_root_metric(matrix, carrier)
    squared = carrier**2 * np.eye(5) - 2j * carrier * matrix
    residual = root_matrix @ root_matrix - squared
    assert np.max(abs(residual)) < 1e-10 * carrier**2
    state = generator.standard_normal(5) + 1j * generator.standard_normal(5)
    flow = 1j * root_matrix @ state
    left = np.conj(flow) @ metric @ state + np.conj(state) @ metric @ flow
    right = -np.conj(state) @ ((squared - np.conj(squared.T)) / (2j * carrier)) @ state
    size = float(np.linalg.norm(state) ** 2)
    root_norm = float(np.linalg.norm(abs(root_matrix)))
    metric_norm = float(np.linalg.norm(abs(metric)))
    squared_norm = float(np.linalg.norm(abs(squared)))
    count = len(state)
    unit = np.finfo(float).eps / 2

    def gamma(operations):
        return operations * unit / (1 - operations * unit)

    identity = (
        size
        * (
            float(np.linalg.norm(residual - np.conj(residual.T), 2))
            + 2 * gamma(count + 1) * (root_norm**2 + squared_norm)
        )
        / (2 * carrier)
    )
    rounding = (
        gamma(4 * count + 8)
        * size
        * (2 * root_norm * metric_norm + squared_norm / carrier)
    )
    assert abs(float(left.real) - float(right.real)) <= identity + rounding
    assert abs(float(left.imag)) <= rounding
    # The identity is not vacuous at this bound: a metric off by one part in 1e6
    # moves the left side by far more than it.
    assert 1e-6 * size * root_norm * metric_norm > 100 * (identity + rounding)


def test_the_square_root_metric_refuses_a_carrier_off_its_domain():
    """A non-positive carrier is refused: the identity is not derived there."""
    operator = wall_operator(block(2), np.array([2.356e15]))
    with pytest.raises(ModeError, match="constant real positive carrier"):
        wall_square_root_metric(operator.generator[:, :, 0], -1.0)


# ------------------------------------------------------------------ admission


def test_out_of_band_bins_are_never_evaluated():
    """The generator is exactly zero off the mask and the limits use only the band."""
    omega = np.linspace(0.0, 4e15, 9)
    active = omega > 2e15
    operator = wall_operator(block(2), omega, active=active)
    assert np.array_equal(operator.active, active)
    assert not np.any(operator.generator[:, :, ~active])
    assert not np.any(operator.coupling[~active])
    assert operator.limits.band == (float(omega[active][0]), float(omega[-1]))


def test_a_sparse_or_mixed_set_is_rejected_by_name():
    """Every incompatible set names what it carries and what the block needs."""
    with pytest.raises(ModeError, match="HE_11 to HE_12"):
        wall_operator(
            [
                MarcatiliMode(125e-6, "He", 1.0, m=1),
                MarcatiliMode(125e-6, "He", 1.0, m=3),
            ],
            np.array([2.356e15]),
        )
    with pytest.raises(ModeError, match="HE_11 to HE_12"):
        wall_operator(
            [
                MarcatiliMode(125e-6, "He", 1.0, m=1),
                MarcatiliMode(125e-6, "He", 1.0, kind="TE", n=0, m=1),
            ],
            np.array([2.356e15]),
        )
    with pytest.raises(ModeError, match="HE_11 to HE_11"):
        wall_operator([RectMode(1e-4, 1e-4, "He", 1.0)], np.array([2.356e15]))
    with pytest.raises(ModeError, match="at least one HE_1m mode"):
        wall_operator([], np.array([2.356e15]))


def test_a_mixed_polarisation_or_model_block_is_rejected_by_name():
    """A scalar block shares one polarisation angle and one capillary model."""
    turned = [
        MarcatiliMode(125e-6, "He", 1.0, m=1),
        MarcatiliMode(125e-6, "He", 1.0, m=2, phi=0.5 * math.pi),
    ]
    with pytest.raises(ModeError, match="one polarisation angle"):
        wall_operator(turned, np.array([2.356e15]))
    mixed = [
        MarcatiliMode(125e-6, "He", 1.0, m=1),
        MarcatiliMode(125e-6, "He", 1.0, m=2, model="reduced"),
    ]
    with pytest.raises(ModeError, match="one capillary model"):
        wall_operator(mixed, np.array([2.356e15]))


def test_a_varying_section_is_built_at_the_position_it_is_asked_for():
    """A taper is a block per position, not a refusal.

    Bound, derived and stated first. The transverse term is `-i u_m^2/(2 k a^2)`, so
    halving the radius quadruples it exactly; the rank-one wall's share of the same
    diagonal entry is `s = -2/(2N - i a k_eff)`, whose modulus is half the recorded
    quasi-static ratio `q = 2N/|a k_eff|`. The ratio of the two ends' diagonals is
    therefore `4(1 + s_narrow)/(1 + s_wide)`, and with `q_narrow = 2 q_wide` its
    departure from 4 is `4|s_narrow - s_wide| = 2(q_narrow - q_wide) = q_narrow` to
    leading order. Measured 1.2015e-2 against a recorded `q_narrow` of 1.2016e-2.
    """
    taper = Linear(125e-6, 62.5e-6, 1e-2)
    tapered = [MarcatiliMode(taper, None, 0.0, m=index + 1) for index in range(2)]
    omega = np.array([2.356e15])
    entrance = wall_operator(tapered, omega, z=0.0)
    exit_block = wall_operator(tapered, omega, z=1e-2)
    assert entrance.limits.radius == 125e-6
    assert exit_block.limits.radius == 62.5e-6
    ratio = exit_block.generator[0, 0, 0] / entrance.generator[0, 0, 0]
    narrow = exit_block.limits.quasi_static
    assert narrow == pytest.approx(2.0 * entrance.limits.quasi_static, rel=1e-12)
    assert abs(ratio - 4.0) == pytest.approx(narrow, rel=1e-3)


def test_a_profile_without_an_analytic_derivative_is_refused_by_name():
    """The connection is a'/a, and a difference of samples is noise, not a slope.

    The connection requires analytic or piecewise derivatives of the declared taper
    profiles; a `FromCallable` without a `slope` reports
    `derivative_is_analytic = False` and is refused rather than differentiated.
    """
    sampled = FromCallable(lambda z: 125e-6 - 6e-3 * z, (0.0, 1e-2))
    modes = [MarcatiliMode(sampled, None, 0.0, m=index + 1) for index in range(2)]
    with pytest.raises(ModeError, match="no analytic derivative"):
        wall_operator(modes, np.array([2.356e15]))


def test_channels_declaring_different_profiles_are_refused_by_name():
    """Two channels that agree at the entrance and nowhere else are two guides.

    The block is built from the lowest channel alone, so the comparison is of the
    declared **profiles** and not of their value at one position: a comparison of
    `a(0)` and `rho(0)` alone would pass two different tapers with the same entrance.
    """
    mine = Linear(125e-6, 62.5e-6, 1e-2)
    theirs = Linear(125e-6, 100e-6, 1e-2)
    mixed = [
        MarcatiliMode(mine, None, 0.0, m=1),
        MarcatiliMode(theirs, None, 0.0, m=2),
    ]
    assert float(mine(np.array([0.0]))[0]) == float(theirs(np.array([0.0]))[0])
    with pytest.raises(ModeError, match="core radius="):
        wall_operator(mixed, np.array([2.356e15]))
    shared = FromCallable(
        lambda z: 125e-6 - 6e-3 * z, (0.0, 1e-2), slope=lambda z: np.full_like(z, -6e-3)
    )
    same = [MarcatiliMode(shared, None, 0.0, m=index + 1) for index in range(2)]
    assert wall_operator(same, np.array([2.356e15])).n_modes == 2
    other = FromCallable(
        lambda z: 125e-6 - 6e-3 * z, (0.0, 1e-2), slope=lambda z: np.full_like(z, -6e-3)
    )
    apart = [
        MarcatiliMode(shared, None, 0.0, m=1),
        MarcatiliMode(other, None, 0.0, m=2),
    ]
    with pytest.raises(ModeError, match="a declared callable at"):
        wall_operator(apart, np.array([2.356e15]))


def test_an_empty_band_or_an_empty_frequency_vector_is_rejected():
    """The formula divides by omega, so a mask with no positive bin is refused."""
    with pytest.raises(ModeError, match="non-empty frequency vector"):
        wall_operator(block(2), np.zeros((0,)))
    with pytest.raises(ModeError, match="no active bin"):
        wall_operator(block(2), np.array([0.0, 1e15]), active=np.array([True, False]))


def test_no_cladding_this_package_ships_can_leave_the_passive_branch():
    """`Re(1/v_HE) >= 0` for every permittivity, which is why the wall always absorbs.

    The algebra in :func:`solprop.modes.capillary._check_wall_passivity` says
    `Re(2s/(s^2+2)) = 2x(|s|^2+2)/|s^2+2|^2` with `x = Re sqrt(e - 1) >= 0` on the
    principal root. This measures it over a grid of real, complex, absorbing and
    below-unity claddings, to an absolute -1e-15 of the value's own scale.
    """
    grid = [0.5, 0.95, 1.0001, 1.3, 1.45, 3.0, 1.45 + 0.05j, 0.6 + 0.2j, 1.0 + 0.5j]
    permittivity = np.array([complex(one) ** 2 for one in grid])
    term, _ = _cladding_term_and_derivative(
        permittivity, np.zeros_like(permittivity), "HE"
    )
    assert np.all(np.real(1.0 / term) >= -1e-15 * abs(1.0 / term))


def test_an_impedance_off_the_passive_branch_is_refused_rather_than_clamped(
    monkeypatch,
):
    """A wall that supplied power would be refused by name, not clamped to zero.

    Unreachable through any cladding a caller can pass -- the test above proves why --
    so the guard is exercised by putting the impedance on the active branch directly.
    It is a defence against a future impedance that does not take the principal root,
    such as the vector averages of the hybrid families, and against a sign slipped
    into this arithmetic.
    """
    principal = capillary._cladding_term_and_derivative

    def active(permittivity, dpermittivity, kind):
        term, derivative = principal(permittivity, dpermittivity, kind)
        return -term, -derivative

    monkeypatch.setattr(capillary, "_cladding_term_and_derivative", active)
    with pytest.raises(ModeError, match="not passive on the branch"):
        wall_operator(block(3, 125e-6, None, 0.0), np.array([2.356e15]))


@pytest.mark.filterwarnings("ignore:The maximum number of subdivisions")
def test_a_validity_ratio_above_its_limit_is_refused_and_below_it_warns():
    """The worst ratio is named in both the refusal and the warning.

    Two cases, because which ratio is worst is itself physics: a narrow core at a
    long wavelength leaves the paraxial regime first (`transverse_ratio`), while a
    wide core with many channels leaves the quasi-static one first
    (`quasi_static`), and each must name itself.
    """
    narrow = block(12, 20e-6, None, 0.0, cladding=1.45)
    long_wave = np.array([2.0 * np.pi * C / 5e-6])
    with pytest.raises(ModeError, match=r"transverse_ratio is 1\.4"):
        wall_operator(narrow, long_wave, validity_limit=0.25)
    with pytest.warns(GrazingIncidenceWarning, match="transverse_ratio"):
        wall_operator(narrow, long_wave, validity_limit=10.0)
    # Which of the two wins is set by 2 N v against u_N, and u_N is about
    # pi (N - 1/4): with this cladding the quasi-static ratio is the larger one
    # only for a few channels, which is the case a wide bore actually runs.
    wide = block(3, 150e-6, None, 0.0, cladding=1.3)
    short_wave = np.array([2.0 * np.pi * C / 800e-9])
    with pytest.raises(ModeError, match="quasi_static"):
        wall_operator(wide, short_wave, validity_limit=0.008)
    with pytest.warns(GrazingIncidenceWarning, match="quasi_static"):
        wall_operator(wide, short_wave, validity_limit=0.07)


@pytest.mark.parametrize("builder", [wall_operator, vector_wall_operator])
def test_the_validity_ratios_are_taken_over_the_scored_bins_alone(builder):
    """A scored mask moves the ratios and the band, never the generator.

    Bound, stated first: **exact** for the generator, which is built on every
    evaluated bin whatever is scored, so it is the unscored block's array (scoring is
    no arithmetic); the ratios are the same per-bin arithmetic restricted to the
    scored bins, so they equal the block built on those bins alone to a few roundings
    (4 eps), and the band is those bins' extremes. A block the
    whole band refuses at the default limit is admitted when only its short-wavelength
    bin is scored: every ratio is proportional to the
    wavelength, so the long bin at three times the wavelength triples each of them
    (0.129 to 0.388 on the scalar block, 0.085 to 0.255 on the vector one).
    """
    modes = block(8, 30e-6, None, 0.0, cladding=1.45)
    if builder is vector_wall_operator:
        modes = vector_block(
            (3, 3), radius=30e-6, gas=None, pressure=0.0, cladding=1.45
        )
    omega = 2.0 * np.pi * C / np.array([3e-6, 2e-6, 1e-6])
    scored = np.array([False, False, True])
    whole = builder(modes, omega, validity_limit=math.inf)
    judged = builder(modes, omega, scored=scored, validity_limit=math.inf)
    alone = builder(modes, omega, active=scored, validity_limit=math.inf)
    assert np.array_equal(judged.generator, whole.generator)
    for name in ("transverse_ratio", "vacuum_ratio", "quasi_static"):
        assert getattr(judged.limits, name) == pytest.approx(
            getattr(alone.limits, name), rel=4 * np.finfo(float).eps, abs=0
        )
    assert judged.limits.band == (float(omega[2]), float(omega[2]))
    assert whole.limits.band == (float(omega[0]), float(omega[2]))
    assert judged.limits.worst < 0.25 < whole.limits.worst
    with pytest.raises(ModeError, match=r"above the 0\.25"):
        builder(modes, omega)
    with pytest.warns(GrazingIncidenceWarning):
        builder(modes, omega, scored=scored)


def test_a_scored_mask_off_the_axis_is_refused_and_an_empty_one_takes_every_bin():
    """A mask of another length is refused; one with no evaluated bin is conservative.

    Bound, stated first: **exact**. Requested limits narrower than one bin leave the
    scored band empty (a coarse study grid can); the ratios are then the
    whole evaluated band's, exactly, so such a grid is never admitted on fewer
    bins than it evaluates.
    """
    modes = block(3)
    omega = np.array([2.0e15, 2.356e15])
    with pytest.raises(ModeError, match="scored mask has shape"):
        wall_operator(modes, omega, scored=np.array([True]))
    empty = wall_operator(modes, omega, scored=np.zeros(2, dtype=bool))
    whole = wall_operator(modes, omega)
    assert empty.limits.worst == whole.limits.worst
    assert empty.limits.band == whole.limits.band


def test_the_limits_report_the_ratio_that_is_worst():
    """`named` and `worst` agree and pick the largest of the three."""
    limits = WallLimits(0.3, 0.1, 0.2, (1e15, 3e15), 4, 125e-6)
    assert limits.named() == ("transverse_ratio", 0.3)
    assert limits.worst == 0.3
    assert "transverse_ratio" in repr(limits)
    assert "u_N/(k a)" in str(limits)


def test_a_declaration_checks_both_of_its_limits():
    """A conditioning below one and a ratio outside (0, 1] are refused by name."""
    with pytest.raises(ModeError, match="at least one"):
        CoherentWall(condition_limit=0.5)
    with pytest.raises(ModeError, match="ratio in"):
        CoherentWall(validity_limit=0.0)
    with pytest.raises(ModeError, match="ratio in"):
        CoherentWall(validity_limit=2.0)
    with pytest.raises(ModeError, match="at least one"):
        CoherentWall(condition_limit=math.inf)


def test_the_operator_prints_without_an_array():
    """The one-line repr names the channels, the active bins and the worst ratio."""
    operator = wall_operator(block(3), np.array([2.356e15, 2.4e15]))
    assert repr(operator).startswith("WallOperator(3 channels, 2 frequencies")
    assert "largest field loss" in str(operator)
    assert repr(CoherentWall()) == (
        "CoherentWall(impedance=mean, condition_limit=10000, "
        "validity_limit=0.25, sections=1)"
    )


def test_the_amplitude_map_is_one_positive_scale_for_every_channel():
    """The orthonormal wall basis and this package's power normalisation differ by a
    channel-independent factor, which is why the generator is the same in both.

    Bound, stated first: `sqrt(N_m) = |J_1(u_m)| a sqrt(pi eps_0 c/2)` and the wall
    basis carries `sqrt(2)/|J_1(u_m)|`, so the ratio is `a sqrt(pi eps_0 c)` for
    every `m` alike, to the 1e-14 of the two square roots. A channel-dependent ratio
    would be a similarity the generator has not been conjugated by.
    """
    radius = 125e-6
    modes = block(5, radius, None, 0.0)
    ratios = [
        math.sqrt(mode.normalisation()) * math.sqrt(2.0) / abs(jv(1, mode.u_nm))
        for mode in modes
    ]
    expected = radius * math.sqrt(math.pi * EPS0 * C)
    assert np.allclose(np.array(ratios, dtype=float), expected, rtol=1e-14)


def test_the_square_root_and_paraxial_generators_agree_where_the_model_is_valid():
    """`i B` minus its carrier is the paraxial generator to `(u_N/(ka))^2/4` relative.

    Bound, stated first: `B = sqrt(k^2 - q^2)` expands as `k - q^2/(2k) - q^4/(8k^3)`,
    so the paraxial generator this package propagates omits a term whose size
    relative to the first correction is `q^2/(4k^2) <= u_N^2/(4(ka)^2)`. That number
    is what production semantics cost, and it is why the transverse ratio is a
    recorded validity limit rather than a comment.
    """
    radius, omega = 125e-6, np.array([2.356e15])
    modes = block(4, radius, None, 0.0)
    operator = wall_operator(modes, omega)
    matrix = operator.generator[:, :, 0]
    carrier = float(np.real(operator.wavenumber[0]))
    root_matrix, _ = wall_square_root_metric(matrix, carrier)
    exact = 1j * (root_matrix - carrier * np.eye(4))
    predicted = modes[-1].u_nm ** 2 / (4.0 * (carrier * radius) ** 2)
    departure = float(np.max(abs(exact - matrix))) / float(np.max(abs(matrix)))
    assert departure == pytest.approx(predicted, rel=0.1)


def test_the_square_root_is_the_principal_branch_of_the_block():
    """`B^2` is `M` itself, and `B`'s spectrum is in the right half plane.

    Bound, stated first. What is checked is the **content**, not the routine that
    produced it, so the root is squared and compared with `M = k^2 I - 2ikG` rather
    than with another call of the same square root on the same input, which could
    not fail. Squaring a computed root costs about `N eps` of `norm(B)^2 = k^2`
    times the root's own conditioning, which for this nearly-`kI` block is one to
    rounding; the gate is `1e-13 k^2`, two orders above that. The **principal**
    branch is the one whose spectrum lies in the right half plane -- what makes
    `W = (B + B^H)/2k` the flux metric of the forward wave and not of its backward
    partner -- and it is asserted on the eigenvalues rather than inferred from the
    construction.
    """
    operator = wall_operator(block(3), np.array([2.356e15]))
    matrix = operator.generator[:, :, 0]
    carrier = float(np.real(operator.wavenumber[0]))
    root_matrix, metric = wall_square_root_metric(matrix, carrier)
    squared = carrier**2 * np.eye(3) - 2j * carrier * matrix
    assert np.max(abs(root_matrix @ root_matrix - squared)) < 1e-13 * carrier**2
    assert np.all(np.real(np.linalg.eigvals(root_matrix)) > 0.0)
    # The metric is the scaled Hermitian part of the root, so it is Hermitian, and
    # close to the identity wherever the paraxial model is the one in use.
    assert np.max(abs(metric - np.conj(metric.T))) < 1e-15 * float(np.max(abs(metric)))
    assert np.max(abs(metric - np.eye(3))) < 1e-4


# --------------------------------------------------------- the taper connection


def _local_basis(orders, radius):
    """Return the orthonormal local basis functions of a block at one core radius."""

    def basis(index, r):
        zero = orders[index]
        return np.sqrt(2.0) * jv(0, zero * r / radius) / (radius * abs(jv(1, zero)))

    return basis


def test_the_connection_is_the_derivative_of_the_implemented_basis():
    """The connection's sign comes from differentiating the profile, not from the note.

    Bound, stated first. The local basis is
    `psi_n(r, z) = sqrt(2) J_0(u_n r/a(z)) / (a(z) |J_1(u_n)|)`, orthonormal under
    `integral_0^a . r dr` at every z, and the connection is
    `C_mn = <psi_m | d psi_n / dz>`. That inner product is taken here with the radius
    differentiated through the **implemented** `Linear` profile by a central
    difference of step 1e-9 m, so the comparison is limited by that difference's own
    `h^2 |a'''|/6` truncation and by the quadrature, which together are below 1e-7
    relative. A sign error would show as a factor of exactly -1 and is what this
    catches.
    """
    taper = Linear(125e-6, 62.5e-6, 1e-2)
    modes = [MarcatiliMode(taper, None, 0.0, m=index + 1) for index in range(3)]
    orders = np.array([mode.u_nm for mode in modes])
    position, step = 5e-3, 1e-9
    radius = float(taper(np.array([position]))[0])
    ahead = _local_basis(orders, float(taper(np.array([position + step])[0:1])[0]))
    behind = _local_basis(orders, float(taper(np.array([position - step])[0:1])[0]))
    here = _local_basis(orders, radius)
    measured = np.array(
        [
            [
                quad(
                    lambda r, m=m, n=n: (
                        here(m, r) * (ahead(n, r) - behind(n, r)) / (2.0 * step) * r
                    ),
                    0.0,
                    radius,
                    limit=400,
                )[0]
                for n in range(3)
            ]
            for m in range(3)
        ]
    )
    formula = wall_connection(modes, position)
    scale = float(np.max(abs(formula)))
    assert_allclose(measured, formula, atol=1e-7 * scale, rtol=1e-6)


def test_the_connection_is_real_skew_and_first_order_in_the_local_slope():
    """C is real and antisymmetric, and every entry scales exactly with a'/a.

    Bound, stated first: `C_mn = (a'/a) 2 s_m s_n u_m u_n/(u_m^2 - u_n^2)` is a fixed
    dimensionless matrix times one scalar, so two profiles with the same `a'/a` at a
    position give the **same** matrix to the last bit, and one with twice the slope
    gives exactly twice it. Antisymmetry is what makes the connection alone norm
    preserving, which is checked on the propagated coefficients below.
    """
    slow = Linear(125e-6, 100e-6, 1e-2)
    modes = [MarcatiliMode(slow, None, 0.0, m=index + 1) for index in range(4)]
    link = wall_connection(modes, 5e-3)
    assert link.dtype == np.float64
    assert np.array_equal(link, -link.T)
    assert np.array_equal(np.diagonal(link), np.zeros(4))
    fast = Linear(125e-6, 75e-6, 1e-2)
    doubled = [MarcatiliMode(fast, None, 0.0, m=index + 1) for index in range(4)]
    radius = float(slow(np.array([5e-3]))[0])
    where = float(np.asarray(fast._stages.positions[0]))
    assert where == 0.0
    # a(z) = 125 - 5 z/L on one and 125 - 10 z/L on the other, so the ratio of the
    # two connections at positions of equal radius is exactly the ratio of slopes.
    at = (125e-6 - radius) / (50e-6) * 1e-2
    assert_allclose(wall_connection(doubled, at), 2.0 * link, rtol=1e-12)


def test_a_pure_change_of_basis_produces_no_conversion():
    """`c = T(z) d` with `-T^-1 T'` reproduces the same physical coefficients.

    Bound, stated first: the transformation is exact arithmetic on a 2x2 unitary, so
    the two propagations agree to the rounding of one ODE solve at `rtol = 1e-12`,
    which is what the tolerance below is. Dropping `-T^-1 T'` instead turns a pure
    relabelling into conversion, and the last assertion measures how much: over a
    quarter turn the final coefficients depart from the right answer by a **measured
    0.59 of their own size**, against a gate of 0.3 declared before the run.

    This is the machinery *any* moving basis requires, and
    the HE1m connection of :func:`wall_connection` is one instance of it.
    """
    generator = np.array([[-1.0 + 2.0j, 0.0], [0.0, -3.0 + 5.0j]])
    turn = 0.5 * math.pi

    def rotation(z):
        angle = turn * z
        return np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )

    def slope(z):
        angle = turn * z
        return turn * np.array(
            [
                [-math.sin(angle), -math.cos(angle)],
                [math.cos(angle), -math.sin(angle)],
            ]
        )

    start = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    direct = solve_ivp(
        lambda z, c: generator @ c, (0.0, 1.0), start, rtol=1e-12, atol=1e-14
    )
    turned = solve_ivp(
        lambda z, d: (
            rotation(z).T @ generator @ rotation(z) @ d - rotation(z).T @ slope(z) @ d
        ),
        (0.0, 1.0),
        start,
        rtol=1e-12,
        atol=1e-14,
    )
    recovered = rotation(1.0) @ turned.y[:, -1]
    scale = float(np.max(abs(direct.y[:, -1])))
    assert_allclose(recovered, direct.y[:, -1], rtol=1e-8, atol=1e-8 * scale)
    dropped = solve_ivp(
        lambda z, d: rotation(z).T @ generator @ rotation(z) @ d,
        (0.0, 1.0),
        start,
        rtol=1e-12,
        atol=1e-14,
    )
    moved = rotation(1.0) @ dropped.y[:, -1]
    departure = float(np.linalg.norm(moved - direct.y[:, -1])) / float(
        np.linalg.norm(direct.y[:, -1])
    )
    assert departure > 0.3, departure


def test_the_connection_alone_conserves_the_coefficient_norm():
    """A real antisymmetric generator is norm preserving, exactly.

    Bound, stated first: `d|c|^2/dz = -c^T (C + C^T) c = 0` identically for a skew C,
    so the only departure is the ODE solver's, here 1e-10 at `rtol = 1e-12` over a
    centimetre of a 2:1 taper.
    """
    taper = Linear(125e-6, 62.5e-6, 1e-2)
    modes = [MarcatiliMode(taper, None, 0.0, m=index + 1) for index in range(4)]
    start = np.zeros(4)
    start[0] = 1.0
    solution = solve_ivp(
        lambda z, c: -wall_connection(modes, float(z)) @ c,
        (0.0, 1e-2),
        start,
        rtol=1e-12,
        atol=1e-14,
    )
    assert float(np.linalg.norm(solution.y[:, -1])) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.slow
def test_the_connection_transports_the_frozen_field_to_second_order():
    """Inside the retained set the connection carries a frozen field, to second order.

    With `c' = -C c` and `C_mn = <psi_m|dz psi_n>` the physical field
    `E = sum_n c_n psi_n` is stationary **within the retained span**: the two terms of
    `dz E` cancel there. What does not cancel is the part of `dz psi_n` outside the
    span, and it does not vanish at the wall while every `psi_m` does, so raising N
    does not remove it -- the omitted coefficients fall only as `1/u_m`.

    Bound, stated first, and the finding: the departure of the propagated
    coefficients from the direct projection of the frozen field onto the local basis
    is **second order in the total contraction** `1 - a_L/a_0`. Halving the
    contraction must therefore quarter it; the predeclared gate is a measured order
    above 1.9, and the three contractions below give 2.008 and 2.004.
    """
    contractions = (0.01, 0.005, 0.0025)
    departures = []
    for contraction in contractions:
        taper = Linear(125e-6, 125e-6 * (1.0 - contraction), 1e-2)
        modes = [MarcatiliMode(taper, None, 0.0, m=index + 1) for index in range(4)]
        orders = np.array([mode.u_nm for mode in modes])
        wide, narrow = 125e-6, 125e-6 * (1.0 - contraction)
        start = np.zeros(4)
        start[0] = 1.0
        solution = solve_ivp(
            lambda z, c, modes=modes: -wall_connection(modes, float(z)) @ c,
            (0.0, 1e-2),
            start,
            rtol=1e-13,
            atol=1e-15,
        )
        here, there = _local_basis(orders, narrow), _local_basis(orders, wide)
        projected = np.array(
            [
                quad(
                    lambda r, m=m, here=here, there=there: here(m, r) * there(0, r) * r,
                    0.0,
                    narrow,
                    limit=400,
                )[0]
                for m in range(4)
            ]
        )
        departures.append(float(np.max(abs(solution.y[:, -1] - projected))))
    orders_measured = [
        math.log2(departures[index] / departures[index + 1]) for index in range(2)
    ]
    assert min(orders_measured) > 1.9, orders_measured


# ------------------------------------------------------------- the vector block
#
# The independent references here are: the numerically integrated angular form of
# the boundary condition, taken over the **implemented** Cartesian fields; complex
# roots of the exact two-family outgoing equation
# `det[diag(T_p, T_{p+2}) - i a Z^-1] = 0`, solved by a root finder that knows
# nothing about the condensation; and this package's own `MarcatiliMode` effective
# index for each of the four families.


def vector_block(
    counts: Sequence[int],
    *,
    orders: Sequence[int] = (1, -1),
    radius: float | ZProfile = 125e-6,
    gas: str | None = "He",
    pressure: float = 1.0,
    kinds: Sequence[ModeKind] | None = None,
    **kwargs,
):
    """Return one azimuthal order of a capillary, `counts` channels per family."""
    if kinds is not None:
        return [
            MarcatiliMode(radius, gas, pressure, kind=kind, n=0, m=index + 1, **kwargs)
            for kind, count in zip(kinds, counts, strict=True)
            for index in range(count)
        ]
    return [
        MarcatiliMode(radius, gas, pressure, n=order, m=index + 1, **kwargs)
        for order, count in zip(orders, counts, strict=True)
        for index in range(count)
    ]


def angular_gram(modes, impedance, z=0.0):
    """Return the boundary form of a mode set, integrated over the implemented field.

    The wall condition is diagonal in the `(r, theta)` components, so the matrix the
    condensation carries is `Z_fg = integral [z_r P_f P_g + z_t Q_f Q_g] dtheta`, with
    `(P, Q)` each channel's own angular pattern at the wall, normalised. The pattern
    is taken from :meth:`MarcatiliMode.field` just inside the wall -- the radial
    factor is common to theta and divides out of a normalised pattern -- so this is a
    quadrature of the field the package implements and not of a formula restated here.
    """
    radius = float(modes[0].core_radius(np.array([z]))[0])
    theta = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    points = np.stack([np.full_like(theta, radius * (1.0 - 1e-6)), theta], axis=-1)
    patterns = []
    for mode in modes:
        components = mode.field(points, z)
        radial = components[:, 0] * np.cos(theta) + components[:, 1] * np.sin(theta)
        tangential = -components[:, 0] * np.sin(theta) + components[:, 1] * np.cos(
            theta
        )
        norm = np.sqrt(np.sum(radial**2 + tangential**2) * (theta[1] - theta[0]))
        patterns.append((radial / norm, tangential / norm))
    weight = theta[1] - theta[0]
    return np.array(
        [
            [
                np.sum(impedance[0] * one[0] * two[0] + impedance[1] * one[1] * two[1])
                * weight
                for two in patterns
            ]
            for one in patterns
        ]
    )


def t_ratio(order, value):
    """Return `T_p(lambda) = q J'_p(q)/J_p(q)` at `lambda = q^2`, exactly."""
    q = np.sqrt(complex(value))
    return order - q * jv(order + 1, q) / jv(order, q)


def vector_root(guess, orders, admittance):
    """Solve `det[diag(T_pA, T_pB) - i H] = 0` near `guess`, independently."""

    def residual(value):
        lam = complex(value[0], value[1])
        block = np.diag([t_ratio(orders[0], lam), t_ratio(orders[1], lam)])
        out = complex(np.linalg.det(block - 1j * admittance))
        return [out.real, out.imag]

    found = root(residual, [guess.real, guess.imag], tol=1e-13)
    assert found.success
    return complex(found.x[0], found.x[1])


def family_loss(kind, order, radius, wavelength, cladding=1.45 + 0j):
    """Return the vector block's decay rate of an isolated family and Marcatili's."""
    omega = np.array([2.0 * np.pi * C / wavelength])
    mode = MarcatiliMode(radius, None, 0.0, kind=kind, n=order, m=1, cladding=cladding)
    block = vector_wall_operator([mode], omega, validity_limit=2.0)
    marcatili = float(np.imag(mode.neff(omega)[0])) * float(omega[0] / C)
    return -float(np.real(block.generator[0, 0, 0])), marcatili, block


def test_the_wall_impedances_are_this_packages_own_transverse_cladding_terms():
    """`impedance` is `(v_TM, v_TE)/k_0`, and their mean is the scalar block's.

    Bound, stated first: **exact** to the rounding of one complex division, 1e-15
    relative. The vector block's two impedances are not a new material model: they
    are the TM and TE Marcatili cladding terms this package already computes for a
    TM_0m and a TE_0m channel, and the scalar block's `1/k_eff = v_HE/k_0` is their
    arithmetic mean, because `v_HE = (v_TE + v_TM)/2` identically. That identity is
    what makes the scalar impedance average a *limit* of this block rather than a
    second model.
    """
    omega = np.array([2.30e15, 2.356e15, 2.40e15])
    modes = vector_block((2, 2))
    block = vector_wall_operator(modes, omega)
    scalar = wall_operator(vector_block((2,), orders=(1,)), omega)
    for index, kind in enumerate(("TM", "TE")):
        expected = np.asarray(
            MarcatiliMode(125e-6, "He", 1.0, kind=kind, n=0)
            .sampled_terms(omega)
            .cladding_term
        ) / (omega / C)
        assert_allclose(block.impedance[index], expected, rtol=1e-15)
    mean = 0.5 * (block.impedance[0] + block.impedance[1])
    assert_allclose(mean, 1.0 / scalar.impedance, rtol=1e-15)
    anisotropy = 0.5 * (block.impedance[0] - block.impedance[1]) / mean
    # (n_r^2 - 1)/(n_r^2 + 1) of silica against a helium fill, which is 0.35 and not
    # a perturbation: this is the size of what the scalar average throws away.
    assert 0.3 < float(np.max(abs(anisotropy))) < 0.4


@pytest.mark.parametrize(
    ("counts", "orders", "kinds"),
    [
        ((2, 2), (1, -1), None),
        ((3, 1), (2, -2), None),
        ((2, 2), None, ("TM", "TE")),
        ((2,), (1,), None),
    ],
)
def test_the_condensed_weight_is_the_integrated_angular_boundary_form(
    counts, orders, kinds
):
    """The wall term is `|h| S |h|` with `S = (C - i a Z^-1)^-1` and `Z` *integrated*.

    Bound, stated first: 1e-12 relative. The wall's own two impedances are diagonal
    in the `(r, theta)` components; what the families see is that matrix integrated
    over their angular patterns, which for `nu >= 1` is `zbar I + delta sigma_x` and
    for `nu = 0` is `diag(1/k_TM, 1/k_TE)`. The quadrature here is of the fields
    :meth:`MarcatiliMode.field` implements, on a uniform theta grid, where the
    trapezoid rule is spectrally exact for a trigonometric polynomial; 1e-12 is the
    conditioning of the 2x2 inverse that follows, far above the quadrature's own
    error.

    Every sign is the implemented field's and none is restated: the traces used here
    are the bare magnitudes `sqrt 2 u_m`, and the pattern the quadrature integrates
    carries whatever sign the mode's own radial profile has at the wall -- including
    the `J_{-p} = (-1)^p J_p` of a hybrid family built from `jv(n - 1, .)`, which
    flips the coupling of every **even** azimuthal order. A sign slipped anywhere in
    the block shows here as a factor of -1.
    """
    omega = np.array([2.30e15, 2.356e15])
    modes = vector_block(counts, orders=orders, kinds=kinds)
    block = vector_wall_operator(modes, omega, validity_limit=math.inf)
    radius = 125e-6
    rows = np.cumsum([0, *[family.n_channels for family in block.families]])[:-1]
    representatives = [modes[index] for index in rows]
    magnitudes = np.sqrt(2.0) * np.array([modes[index].u_nm for index in rows])
    truncation = np.diag([float(family.truncation) for family in block.families])
    for bin_index in range(omega.size):
        gram = angular_gram(representatives, block.impedance[:, bin_index])
        condensed = np.linalg.inv(truncation - 1j * radius * np.linalg.inv(gram))
        carrier = complex(block.wavenumber[bin_index])
        scale = 1.0 / (2.0 * carrier * radius**2)
        predicted = 1j * scale * np.outer(magnitudes, magnitudes) * condensed
        transverse = -1j * scale * np.array([mode.u_nm**2 for mode in representatives])
        measured = block.generator[np.ix_(rows, rows)][:, :, bin_index] - np.diag(
            transverse
        )
        # The atol is the same 1e-12 of the block's own scale: the nu = 0 families
        # are exactly decoupled, and a relative bound cannot be taken on a zero.
        assert_allclose(
            measured,
            predicted,
            rtol=1e-12,
            atol=1e-12 * float(np.max(abs(measured))),
        )
    assert [family.n_channels for family in block.families] == list(counts)


def test_the_family_traces_are_the_implemented_basis_derivative_at_the_wall():
    """Each column of `family_traces` is `rho_m'(1)` of the mode's own profile.

    Bound, stated first: 1e-4 relative, the first-order error of a one-sided
    difference of step 1e-6 on a quantity of order ten -- the same reference the
    scalar block uses, here taken family by family and including the sign.
    The orthonormal radial function is `rho_m(x) = sqrt 2 jv(n-1, u_m x)/|J_p+1(u_m)|`
    with `jv`, not `J_{|p|}`: that is what `MarcatiliMode.field` builds and therefore
    what the coefficients the block propagates are coefficients of.
    """
    modes = vector_block((3, 3), orders=(2, -2))
    block = vector_wall_operator(modes, np.array([2.356e15]))
    step = 1e-6
    for index, mode in enumerate(modes):
        order = mode.n - 1
        zero = mode.u_nm
        scale = np.sqrt(2.0) / abs(jv(abs(order) + 1, zero))
        difference = scale * (jv(order, zero) - jv(order, zero * (1.0 - step))) / step
        column = int(np.flatnonzero(block.family_traces[index])[0])
        assert_allclose(block.family_traces[index, column], difference, rtol=1e-4)
    assert [family.radial_sign for family in block.families] == [1.0, -1.0]


def test_the_angular_form_couples_only_the_two_families_of_one_order():
    """The wall's selection rule: `delta` to the partner, zero to everything else.

    Bound, stated first: 1e-12 of the mean impedance, the quadrature's own floor. The
    anisotropy is a `cos 2 theta` operator, so it can only move an angular harmonic by
    two -- which is exactly the step from the `J_{nu-1}` family of an order to its
    `J_{nu+1}` partner. Every other pairing integrates to zero, including the
    partner's **orthogonal polarisation**, which is what makes one polarisation angle
    a closed block rather than half of a four-family one.
    """
    impedance = np.array([0.37, 1.13])
    delta = 0.5 * (impedance[0] - impedance[1])
    mean = 0.5 * (impedance[0] + impedance[1])
    he = MarcatiliMode(125e-6, "He", 1.0, n=1, m=1)
    partners = {
        "the EH_1m partner": MarcatiliMode(125e-6, "He", 1.0, n=-1, m=1),
        "its orthogonal polarisation": MarcatiliMode(
            125e-6, "He", 1.0, n=-1, m=1
        ).orthogonal_polarisation,
        "HE_2m": MarcatiliMode(125e-6, "He", 1.0, n=2, m=1),
        "the J_3 family of nu = 2": MarcatiliMode(125e-6, "He", 1.0, n=-2, m=1),
        "TE_0m": MarcatiliMode(125e-6, "He", 1.0, kind="TE", n=0, m=1),
        "TM_0m": MarcatiliMode(125e-6, "He", 1.0, kind="TM", n=0, m=1),
    }
    wanted = {
        "the EH_1m partner": delta,
        "its orthogonal polarisation": 0.0,
        "HE_2m": 0.0,
        "the J_3 family of nu = 2": 0.0,
        "TE_0m": 0.0,
        "TM_0m": 0.0,
    }
    for name, partner in partners.items():
        gram = angular_gram([he, partner], impedance)
        assert abs(gram[0, 1] - wanted[name]) < 1e-12 * mean, name
        assert abs(gram[0, 0] - mean) < 1e-12 * mean, name


@pytest.mark.parametrize("count", [4, 8, 16])
def test_the_vector_eigenvalues_converge_to_independent_two_family_roots(count):
    """The lowest transverse eigenvalues approach the exact vector roots as 1/N.

    Bound, stated first: the condensation's quasi-static error falls like 1/N, so the
    relative departure of each of the six lowest eigenvalues from the independently
    solved root of `det[diag(T_0, T_2) - i a Z^-1] = 0` is below `2e-4/N` and at
    least 1.8 times smaller when `N` doubles. The reference is the **coupled** root
    equation: it is not the scalar one twice, and its roots are not the union of two
    single-family sets.
    """
    radius, wavelength, cladding = 150e-6, 800e-9, 1.3
    omega = np.array([2.0 * np.pi * C / wavelength])
    modes = vector_block(
        (count, count), radius=radius, gas=None, pressure=0.0, cladding=cladding
    )
    block = vector_wall_operator(modes, omega, validity_limit=10.0)
    carrier = complex(block.wavenumber[0])
    matrix = block.generator[:, :, 0] * (2.0 * carrier * radius**2) / 1j
    eigenvalues = np.sort_complex(-np.linalg.eigvals(matrix))
    mean = 0.5 * complex(block.impedance[0, 0] + block.impedance[1, 0])
    difference = 0.5 * complex(block.impedance[0, 0] - block.impedance[1, 0])
    impedance = np.array([[mean, difference], [difference, mean]])
    admittance = radius * np.linalg.inv(impedance)
    guesses = np.concatenate([jn_zeros(0, 3) ** 2, jn_zeros(2, 3) ** 2])
    exact = np.array(
        [vector_root(g * (1 - 2j * mean / radius), (0, 2), admittance) for g in guesses]
    )
    taken = np.array([eigenvalues[np.argmin(abs(eigenvalues - e))] for e in exact])
    departure = np.max(abs(taken - exact) / abs(exact))
    assert departure < 2e-4 / count, departure


def test_the_truncation_constant_carries_each_familys_bessel_order():
    """`2 N_f + p_f`, not `2 N_f`: without `p_f` the eigenvalues stop converging.

    Bound, stated first: the departure from the exact single-family roots falls like
    1/N with the Bessel order in the constant and **stalls** without it. The `p` comes
    from the exact recurrence `q J_{p+1}(q) = (p - i eta) J_p(q)` and is invisible for
    the `J_0` family of `HE_1m`, where `p = 0` -- which is why the scalar block
    never needed it and why it has to be derived here rather than carried
    over. Measured at `p = 2`: with the order, 2.0e-4 falling to 1.2e-5 from N = 5 to
    N = 80; without it, 1.6e-4 stalling at 4.2e-5.
    """
    eta = 300.0 + 40.0j
    order = 2
    exact = np.array([robin_root_of_order(order, index, eta) ** 2 for index in (1, 2)])
    departures = {True: [], False: []}
    for with_order in (True, False):
        for count in (5, 10, 20, 40, 80):
            zeros = jn_zeros(order, count)
            traces = -np.sqrt(2.0) * np.sign(jv(order + 1, zeros)) * zeros
            constant = 2 * count + (order if with_order else 0)
            condensed = -np.diag(zeros**2) + np.outer(traces, traces) / (
                constant - 1j * eta
            )
            values = np.sort_complex(-np.linalg.eigvals(condensed))
            departures[with_order].append(
                float(np.max(abs(values[:2] - exact) / abs(exact)))
            )
    with_order, without = departures[True], departures[False]
    assert with_order[-1] < 0.5 * with_order[-2] * 1.2, with_order
    assert with_order[-1] < 0.4 * without[-1], (with_order, without)
    assert without[-1] > 0.9 * without[-2], without
    # and the block's own families declare that constant
    families = vector_wall_operator(vector_block((2, 3)), np.array([2.356e15])).families
    assert [family.truncation for family in families] == [4, 8]
    assert [family.bessel_order for family in families] == [0, 2]


@pytest.mark.parametrize(
    ("kind", "order"),
    [("TE", 0), ("TM", 0), ("HE", 1), ("HE", -1), ("HE", 2), ("HE", -2)],
)
@pytest.mark.parametrize(("radius", "wavelength"), [(125e-6, 800e-9), (50e-6, 1e-6)])
def test_every_family_reduces_to_its_own_marcatili_loss(
    kind, order, radius, wavelength
):
    """With N = 1 each family's decay is this package's Marcatili loss for its `v_n`.

    Bound, derived and stated first: the condensed diagonal's departure from
    `(omega/c) Im n_eff` is `-(c^2 t^2 + u^2/(2 (k a)^2))` with `c = 2 + p` the
    family's truncation constant and `t = |v_n|/(k_0 a)`, exactly as derived
    for `HE_1m`, and agreement is required to **1 %** of that prediction. This is the
    reduction the vector block turns on: TE_0m sees `v_TE = 1/sqrt(e-1)`, TM_0m
    sees `v_TM = e/sqrt(e-1)` -- a factor `e` apart, which no impedance-averaged block
    can produce -- and the two hybrid families of an order both see the average
    `v_HE`, each with its own Bessel order in the constant.
    """
    measured, marcatili, block = family_loss(kind, order, radius, wavelength)
    family = block.families[0]
    zero = float(jn_zeros(family.bessel_order, 1)[-1])
    omega = np.array([2.0 * np.pi * C / wavelength])
    cladding = complex(
        MarcatiliMode(radius, None, 0.0, kind=kind, n=order, m=1, cladding=1.45 + 0j)
        .sampled_terms(omega)
        .cladding_term[0]
    )
    scale = abs(cladding) / (float(omega[0] / C) * radius)
    predicted = -(
        family.truncation**2 * scale**2
        + zero**2 / (2.0 * (float(omega[0] / C) * radius) ** 2)
    )
    assert abs(measured / marcatili - 1.0 - predicted) < 0.01 * abs(predicted)


VECTOR_FAMILIES: dict[str, tuple[ModeKind, int]] = {
    "TE": ("TE", 0),
    "TM": ("TM", 0),
    "HE": ("HE", 1),
    "EH": ("HE", -1),
}
"""How the independent prototype's family names map onto this package's mode kinds."""


@pytest.mark.medium
def test_the_block_reproduces_the_prototype_vector_boundary_roots():
    """Each family's loss is the **exact** outgoing vector root's, to a derived bound.

    The reference is the committed prototype record `vector_roots.json`, whose
    rows are the exact outgoing Maxwell boundary problem of a 150 micrometre bore --
    tangential `E` and `H` matched to outgoing Hankel functions in an infinite
    cladding, with the hybrid cross term -- solved by a prototype that knows nothing
    about any condensation. It is the vector boundary reference
    the vector block is validated against.

    Bound, stated first and in **two** derived parts: the block departs from the exact
    root by its own condensation error, `-(c^2 t^2 + u^2/(2 (k a)^2))` with
    `c = 2 + p` the family's truncation constant, **plus** this package's own
    Marcatili expansion's departure from the same root, which the receipt lets us
    measure rather than assume. Their sum is required to 3 % of itself at every
    family, wavelength and cladding of the receipt -- eight geometries and four
    families each -- and the two terms are comparable, so neither can be hiding the
    other. Measured worst 0.7 %, at 12 micrometres, where both terms are largest.

    The blocks here are built with a `validity_limit` of 2, which admits the
    receipt's 12 micrometre rows: they are marginal by design, which is what makes the
    departure measurable at all, and the gating of those ratios is a test of its own.
    """
    rows = json.loads(VECTOR_ROOTS.read_text())
    assert len(rows) == 8
    radius = 150e-6
    for row in rows:
        omega = np.array([2.0 * np.pi * C / row["wavelength"]])
        cladding = complex(row["clad"])
        for exact_root in row["roots"]:
            kind, order = VECTOR_FAMILIES[exact_root["family"]]
            mode = MarcatiliMode(
                radius, None, 0.0, kind=kind, n=order, m=1, cladding=cladding
            )
            block = vector_wall_operator([mode], omega, validity_limit=2.0)
            measured = -float(np.real(block.generator[0, 0, 0]))
            exact = exact_root["attenuation_radius"] / radius
            marcatili = float(np.imag(mode.neff(omega)[0])) * float(omega[0] / C)
            scale = abs(complex(mode.sampled_terms(omega).cladding_term[0])) / (
                float(omega[0] / C) * radius
            )
            condensation = -(
                (block.families[0].truncation * scale) ** 2
                + mode.u_nm**2 / (2.0 * (float(omega[0] / C) * radius) ** 2)
            )
            predicted = condensation + (marcatili / exact - 1.0)
            departure = measured / exact - 1.0
            assert abs(departure - predicted) < 0.03 * abs(predicted), (
                row["wavelength"],
                row["clad"],
                exact_root["family"],
                departure,
                predicted,
            )
        # The prototype's own mixed-family angular Gram, in v units, is this block's
        # impedance matrix: the same two eigenvalues and the same off-diagonal.
        pair = [
            MarcatiliMode(radius, None, 0.0, n=n, m=1, cladding=cladding)
            for n in (1, -1)
        ]
        block = vector_wall_operator(pair, omega, validity_limit=2.0)
        impedance = block.impedance[:, 0] * float(omega[0] / C)
        mean = 0.5 * (impedance[0] + impedance[1])
        delta = 0.5 * (impedance[0] - impedance[1])
        assert_allclose(
            np.array([[mean, delta], [delta, mean]]),
            np.asarray(row["wall_gram"]),
            rtol=1e-12,
        )
        assert_allclose(
            np.sort(np.real(impedance)),
            np.asarray(row["wall_eigenvalues"]),
            rtol=1e-12,
        )
        # and the prototype's own rotation check, which this block inherits by being
        # the same form: its Gram does not move when the reference angle does.
        assert row["angular_rotation_residual"] < 1e-14


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.GrazingIncidenceWarning")
@pytest.mark.parametrize(("wavelength", "clad"), [(0.8e-6, 1.45), (6e-6, 1.45)])
@pytest.mark.parametrize("count", [4, 32])
def test_the_mixed_family_coupling_moves_the_eigenvalue_towards_the_exact_root(
    wavelength, clad, count, monkeypatch
):
    """The cross term is physics: it takes `EH_11` **towards** the exact Maxwell root.

    Everything else here that touches `delta` is either the grazing-impedance model
    itself or a ratio blind to it; the prototype's roots are the exact `nu = 1`
    outgoing Maxwell problem **with** the hybrid cross term, so they resolve the
    coupling. Bound, stated first: with the two families' impedances rotated into
    the block, the eigenvalue nearest `EH_11`
    departs from the exact root by less than **half** of what the same block with the
    impedance averaged (`delta = 0`, two copies of the scalar model) departs by, at
    four and at thirty-two channels per family (the thirty-two-channel block is
    marginal by the paraxial ratios, which is what makes the departure measurable,
    and its grazing warning is expected). Measured: the coupling cuts the departure
    by three to six times at every count and wavelength.
    """
    rows = json.loads(VECTOR_ROOTS.read_text())
    row = next(
        one
        for one in rows
        if abs(one["wavelength"] - wavelength) < 1e-12
        and abs(one["clad"] - clad) < 1e-9
    )
    exact = next(one for one in row["roots"] if one["family"] == "EH")
    radius = 150e-6
    omega = np.array([2.0 * np.pi * C / wavelength])
    cladding = complex(clad)
    modes = [
        MarcatiliMode(radius, None, 0.0, kind="HE", n=order, m=index, cladding=cladding)
        for order in (1, -1)
        for index in range(1, count + 1)
    ]
    eh_11 = count  # the first channel of the second family

    def nearest_departure(block):
        eigenvalues = np.linalg.eigvals(block.generator[:, :, 0])
        own = block.generator[eh_11, eh_11, 0]
        nearest = eigenvalues[np.argmin(abs(eigenvalues - own))]
        return -float(np.real(nearest)) / (exact["attenuation_radius"] / radius) - 1.0

    coupled = nearest_departure(vector_wall_operator(modes, omega, validity_limit=2.0))
    true_impedances = capillary_module._vector_impedances

    def averaged(mode, band):
        both = true_impedances(mode, band)
        return np.repeat(np.mean(both, axis=0, keepdims=True), 2, axis=0)

    monkeypatch.setattr(capillary_module, "_vector_impedances", averaged)
    uncoupled = nearest_departure(
        vector_wall_operator(modes, omega, validity_limit=2.0)
    )
    assert abs(coupled) < 0.5 * abs(uncoupled), (count, wavelength, coupled, uncoupled)


def test_the_hybrid_families_take_the_impedance_average_and_not_the_admittance_one():
    """The averaged quantity is `1/k`, not `k`, and the difference is 13 %.

    Bound, stated first: averaging the *admittances* instead gives a loss too low by
    `4 n_r^2/(1 + n_r^2)^2` -- 0.870 for a silica wall at 800 nm -- and the measured
    ratio of the two must land on that factor to 1e-6. It is the documented pitfall
    of the supplied wall note (PRE 87 Eq. 15 averages the discrete boundary values),
    and the vector block is where it is decided rather than assumed: the average
    falls out of `Z = zbar I + delta sigma_x` with `Z` an **impedance** matrix,
    because the condensation is linear in `1/k` near a Dirichlet wall.
    """
    omega = np.array([2.0 * np.pi * C / 800e-9])
    block = vector_wall_operator(
        [MarcatiliMode(125e-6, None, 0.0, n=1, m=1, cladding=1.45 + 0j)], omega
    )
    impedance = block.impedance[:, 0]
    mean = 0.5 * (impedance[0] + impedance[1])
    admittance_mean = 2.0 / (1.0 / impedance[0] + 1.0 / impedance[1])
    ratio = float(abs(admittance_mean / mean))
    contrast = 1.45**2
    assert abs(ratio - 4.0 * contrast / (1.0 + contrast) ** 2) < 1e-6
    # and the block took the impedance average: its loss is the package's own.
    measured, marcatili, _ = family_loss("HE", 1, 125e-6, 800e-9)
    assert abs(measured / marcatili - 1.0) < 1e-4


def test_one_hybrid_family_alone_is_the_scalar_block():
    """Dropping the partner family gives back the scalar block, to 1e-15.

    Bound, stated first: 1e-15 relative, the rounding of the two different
    contractions -- `outer(h, h) * g` there and `einsum(V, W, V)` here. The reduction
    the supplied note asks for is exact rather than asymptotic, because the `HE_1m`
    family's Bessel order is zero, so its truncation constant `2N + p` *is* `2N` and
    its diagonal impedance *is* the impedance average. The scalar model is therefore
    a restriction of this one and not a competing model -- and the rounding-level
    difference is why it keeps its own function, so that a declared scalar wall
    propagates what it did before.
    """
    omega = np.array([2.30e15, 2.356e15, 2.40e15])
    modes = vector_block((4,), orders=(1,))
    scalar = wall_operator(modes, omega)
    vector = vector_wall_operator(modes, omega)
    assert_allclose(vector.generator, scalar.generator, rtol=1e-15, atol=0.0)
    assert vector.families[0].truncation == 8


def test_the_vector_block_is_covariant_under_a_rotation_of_the_launch():
    """A rotated launch rotates the answer: the generator does not move at all.

    Bound, stated first: **exact**, structurally. The wall is rotationally symmetric,
    so turning every channel of a block by one angle turns the physical field by that
    angle and leaves the coefficients alone; in `phi` of ADR-031 the generator must
    therefore be the same array at every angle. The physical content is checked beside
    it: the off-diagonal of the integrated angular form is `delta cos(nu (phi_A -
    phi_B))`, so it is the **common** angle that drops out and a relative one that does
    not -- a block whose families disagree about `phi` is a different operator, and is
    refused at the door
    (`test_a_vector_block_refuses_what_it_would_otherwise_override`).
    """
    omega = np.array([2.30e15, 2.356e15])
    reference = vector_wall_operator(vector_block((3, 3), phi=0.0), omega)
    for angle in (0.3, np.pi / 4, np.pi / 2, -1.1):
        turned = vector_wall_operator(vector_block((3, 3), phi=angle), omega)
        assert np.array_equal(turned.generator, reference.generator)
    # The exact half above is structural -- `phi` enters no arithmetic of the block
    # and is only compared for equality at the door -- so the physical content is the
    # angular form, taken at nu = 1 and at nu = 2, where `cos(2 offset)` is a different
    # function of the offset and the second family's `radial_sign`, (-1)^(nu + 1),
    # flips the coupling's sign.
    impedance = np.array([0.37, 1.13])
    delta = 0.5 * (impedance[0] - impedance[1])
    for order in (1, 2):
        for offset in (0.0, 0.3, np.pi / 4, np.pi / 2):
            pair = [
                MarcatiliMode(125e-6, "He", 1.0, n=order, m=1),
                MarcatiliMode(125e-6, "He", 1.0, n=-order, m=1, phi=offset),
            ]
            gram = angular_gram(pair, impedance)
            expected = (-1) ** (order + 1) * delta * np.cos(order * offset)
            assert abs(gram[0, 1] - expected) < 1e-12, (order, offset)


def test_the_vector_block_is_passive_and_its_hermitian_part_has_rank_two():
    """Passivity is `Re k_TE >= 0` and `Re k_TM >= 0`, and the wall term is rank two.

    Bound, stated first: every Hermitian eigenvalue below `64 N eps` of the block's
    own scale, and the numerical rank of the Hermitian part exactly the number of
    families. The algebra: with `S = (C - i H)^-1` and `H = a Z^-1`,
    `(S - S^H)/2i = S (H + H^H)/2 S^H`, so the wall term's Hermitian part is negative
    semidefinite exactly when the Hermitian part of the **admittance** is positive
    semidefinite, which is `Re k_TE >= 0` and `Re k_TM >= 0` in the wall's own
    components. That is the vector generalisation of the rank-one statement of the
    scalar block, and it is why the guard kept unreachable there is the right guard.
    """
    omega = np.array([2.30e15, 2.356e15, 2.40e15])
    block = vector_wall_operator(vector_block((3, 2)), omega)
    moved = np.moveaxis(block.generator[:, :, block.active], (0, 1), (-2, -1))
    hermitian = 0.5 * (moved + np.conj(np.swapaxes(moved, -1, -2)))
    values = np.linalg.eigvalsh(hermitian)
    scale = float(np.max(abs(block.generator)))
    assert float(np.max(values)) < 64.0 * 5 * np.finfo(float).eps * scale
    assert all(np.linalg.matrix_rank(part, tol=1e-9 * scale) == 2 for part in hermitian)


def test_no_cladding_can_take_either_vector_impedance_off_the_passive_branch():
    """Both wall impedances stay passive for every complex permittivity.

    The vector counterpart of the scalar block's proof, and the reason the passivity
    guard stays defence in depth rather than a reachable refusal. With
    `s = sqrt(e - 1)` on the principal root, `k_TE/k_0 = s` has `Re s >= 0` by that
    branch, and `k_TM/k_0 = s/e` has `Re(s bar e)/|e|^2`, which for `Re e < 0` and
    `Im e = b > 0` is `b (|Re e| + 2)/(2 sqrt(|Re e| + 1) |e|^2) > 0`. Swept here
    over 1001 x 401 permittivities across `Re e` in [-50, 50] and `Im e` in [-5, 5],
    gain and loss alike: the smallest real part found is exactly zero.
    """
    real = np.linspace(-50.0, 50.0, 1001)
    imaginary = np.linspace(-5.0, 5.0, 401)
    permittivity = real[:, None] + 1j * imaginary[None, :]
    root = np.sqrt(permittivity - 1.0 + 0j)
    tangential = root
    with np.errstate(divide="ignore", invalid="ignore"):
        radial = root / permittivity
    finite = np.isfinite(radial)
    assert float(np.min(np.real(tangential))) >= 0.0
    assert float(np.min(np.real(radial)[finite])) >= 0.0


@pytest.mark.parametrize(
    ("modes", "message"),
    [
        ([], "at least one capillary mode"),
        ([RectMode(10e-6, 10e-6)], "is a RectMode"),
        (
            [
                MarcatiliMode(125e-6, "He", 1.0, n=n, m=m + 1)
                for n, m in ((1, 0), (1, 2))
            ],
            "contiguous radial set",
        ),
        (
            [MarcatiliMode(125e-6, "He", 1.0, n=n, m=1) for n in (1, -1, 2)],
            "at most two radial families",
        ),
        (
            [MarcatiliMode(125e-6, "He", 1.0, n=n, m=1) for n in (1, 2)],
            "and no others",
        ),
        (
            [
                MarcatiliMode(125e-6, "He", 1.0, kind="TE", n=0, m=1),
                MarcatiliMode(125e-6, "He", 1.0, n=1, m=1),
            ],
            "mixes the transverse order",
        ),
    ],
)
def test_a_set_that_is_not_one_azimuthal_order_is_refused_by_name(modes, message):
    """Every set the vector block is not derived for is named, not adapted.

    The wall is exactly diagonal in the total azimuthal order, so a set spanning two
    orders is two blocks and not one; each family's boundary self term counts its own
    retained radial orders, so a gap in one is a trace with nowhere to go; and a
    structured cladding is not a Robin condition on a bare capillary at all.
    """
    with pytest.raises(ModeError, match=message):
        vector_wall_operator(modes, np.array([2.356e15]))


def test_a_vector_block_refuses_what_it_would_otherwise_override():
    """`loss=`, `model=`, a mixed guide, a mixed angle and a lattice, each by name."""
    omega = np.array([2.356e15])
    for overridden, message in (
        (vector_block((1, 1), loss=False), "is the loss model"),
        (vector_block((1, 1), loss=0.5), "is the loss model"),
        (vector_block((1, 1), model="reduced"), "one capillary model"),
    ):
        with pytest.raises(ModeError, match=message):
            vector_wall_operator(overridden, omega)
    mixed = [
        MarcatiliMode(125e-6, "He", 1.0, n=1, m=1),
        MarcatiliMode(125e-6, "He", 1.0, n=-1, m=1, phi=0.3),
    ]
    with pytest.raises(ModeError, match="one polarisation angle"):
        vector_wall_operator(mixed, omega)
    clad = [
        MarcatiliMode(125e-6, "He", 1.0, n=1, m=1, cladding=1.45 + 0j),
        MarcatiliMode(125e-6, "He", 1.0, n=-1, m=1, cladding=2.6 + 0.1j),
    ]
    with pytest.raises(ModeError, match="samples a different cladding"):
        vector_wall_operator(clad, omega)
    lattice = [
        VincettiMode(15e-6, "He", 1.0, lattice=TubeLattice(180e-9, 9e-6, 6), m=1)
    ]
    with pytest.raises(ModeError, match="is a VincettiMode"):
        vector_wall_operator(lattice, omega)


def test_a_vector_block_checks_its_band_and_its_ratios_like_the_scalar_one():
    """An empty band, an empty vector and a ratio above the limit are each named."""
    omega = np.array([2.356e15, 2.4e15])
    modes = vector_block((2, 2))
    with pytest.raises(ModeError, match="non-empty frequency vector"):
        vector_wall_operator(modes, np.array([]))
    with pytest.raises(ModeError, match="no active bin"):
        vector_wall_operator(modes, omega, active=np.zeros(2, dtype=bool))
    out_of_band = vector_wall_operator(modes, omega, active=np.array([True, False]))
    assert np.all(out_of_band.generator[:, :, 1] == 0.0)
    assert np.all(out_of_band.weight[:, :, 1] == 0.0)
    narrow = vector_block((3, 3), radius=150e-6, gas=None, pressure=0.0, cladding=1.3)
    with pytest.raises(ModeError, match="quasi_static"):
        vector_wall_operator(narrow, np.array([2.356e15]), validity_limit=0.008)
    with pytest.warns(GrazingIncidenceWarning, match="quasi_static"):
        vector_wall_operator(narrow, np.array([2.356e15]), validity_limit=0.09)


def test_the_validity_ratios_take_the_highest_order_of_every_family():
    """`u_N` is the largest retained zero, which need not be the last channel's.

    Bound, stated first: **exact**, to the rounding of one division. A vector block
    may retain more radial orders of one family than of the other -- three `HE_1m` and
    one `EH_1m`, say -- and then the channel order ends on `j_{2,1} = 5.14` while the
    block actually reaches `j_{0,3} = 8.65`. Reading the last channel would understate
    both paraxial ratios by 1.7 and could admit a block outside the model's range; the
    scan is over every family. The quasi-static ratio is the other kind of maximum,
    over the families' truncation constants and their own row of the impedance.
    """
    omega = np.array([2.356e15])
    modes = vector_block((3, 1))
    block = vector_wall_operator(modes, omega)
    zeros = np.array([mode.u_nm for mode in modes])
    assert zeros[-1] < zeros.max()
    carrier = complex(block.wavenumber[0])
    assert block.limits.transverse_ratio == pytest.approx(
        float(zeros.max() / (abs(carrier) * 125e-6)), rel=1e-15
    )
    assert block.limits.vacuum_ratio == pytest.approx(
        float(zeros.max() / (float(omega[0] / C) * 125e-6)), rel=1e-15
    )
    truncations = [family.truncation for family in block.families]
    assert truncations == [2 * 3 + 0, 2 * 1 + 2]
    mean = 0.5 * complex(block.impedance[0, 0] + block.impedance[1, 0])
    delta = 0.5 * complex(block.impedance[0, 0] - block.impedance[1, 0])
    rows = (abs(mean) + abs(delta)) / 125e-6
    assert block.limits.quasi_static == pytest.approx(
        max(truncations) * rows, rel=1e-12
    )


def test_the_vector_operator_prints_without_an_array():
    """The one-line repr names the order, its families and the worst ratio."""
    block = vector_wall_operator(vector_block((2, 1)), np.array([2.356e15, 2.4e15]))
    assert repr(block).startswith("VectorWallOperator(nu=1, HE_1m+EH_1m, 3 channels")
    assert "wall anisotropy" in str(block)
    assert block.n_modes == 3
    assert block.n_families == 2
    assert repr(CoherentWall(impedance="te_tm")).startswith(
        "CoherentWall(impedance=te_tm"
    )
    # Set after construction, because the runtime type hook this session installs
    # refuses an unknown Literal at the constructor and production installs no such
    # hook: the declaration's own check is what a script meets.
    unknown = CoherentWall()
    object.__setattr__(unknown, "impedance", "tensor")
    with pytest.raises(ModeError, match=r"CoherentWall\.impedance"):
        unknown.__post_init__()


def test_an_eh_channel_can_be_labelled_and_collected():
    """A negative azimuthal index is the EH family, and its label says so.

    A regression: :class:`MarcatiliMode` has always built a
    hybrid of negative `n` -- the `J_{|n|+1}` family of the order -- and its own
    docstring says so, but the channel label passed `n` through to a field that is
    non-negative by contract, so such a mode raised as soon as it was put in a
    :class:`~solprop.modes.collection.ModeCollection` and could not be propagated at
    all. It is labelled like the exact step-index channel of the same physical mode.
    """
    mode = MarcatiliMode(125e-6, "He", 1.0, n=-1, m=2)
    label = mode.channel_label
    assert (label.family, label.azimuthal_order, label.radial_order) == ("EH", 1, 2)
    assert label.display_label == "EH1,2:even"
    partner = mode.orthogonal_polarisation
    assert partner is not None
    assert partner.channel_label.display_label == "EH1,2:odd"
    collection = ModeCollection(tuple(vector_block((2, 2))))
    assert [mode.channel_label.display_label for mode in collection.modes] == [
        "HE1,1:even",
        "HE1,2:even",
        "EH1,1:even",
        "EH1,2:even",
    ]


# ------------------------------------------- the vector block on a varying guide


def implemented_connection(modes, profile, z, *, relative=1e-5, n_r=400, n_theta=64):
    """Return `<psi_m|d_z psi_n>` of a mode set by quadrature of its implemented fields.

    Nothing of the connection's formula is in it: each channel's field is
    :meth:`MarcatiliMode.field`, normalised at every `z` by its own closed-form
    :meth:`MarcatiliMode.normalisation`, differentiated along the taper by a
    fourth-order central difference whose step moves the radius by `relative` of
    itself, and paired over the cross-section by Gauss-Legendre in `r` on `[0, a(z)]`
    and the periodic trapezoid in `theta`. Also returns the Gram of the fields at `z`.
    """
    radius = float(profile(np.array([z]))[0])
    slope = float(profile.derivative(np.array([z]))[0])
    step = relative * radius / abs(slope)
    nodes, weights = np.polynomial.legendre.leggauss(n_r)
    r = 0.5 * radius * (nodes + 1.0)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    points = np.stack([rr.ravel(), tt.ravel()], axis=-1)
    measure = (0.5 * radius * weights[:, None] * rr * (2.0 * np.pi / n_theta)).ravel()

    def normed(mode, position):
        scale = math.sqrt(2.0 * mode.normalisation(position) / (EPS0 * C))
        return mode.field(points, position) / scale

    here = [normed(mode, z) for mode in modes]
    slopes = [
        (
            8.0 * (normed(mode, z + step / 2) - normed(mode, z - step / 2))
            - (normed(mode, z + step) - normed(mode, z - step))
        )
        / (6.0 * step)
        for mode in modes
    ]

    def pair(first, second):
        return float(np.sum(measure * np.sum(first * second, axis=-1)))

    gram = np.array([[pair(one, two) for two in here] for one in here])
    link = np.array([[pair(one, two) for two in slopes] for one in here])
    return link, gram


VECTOR_ORDERS = (
    pytest.param({"orders": (1, -1)}, 0.0, id="HE_1m+EH_1m"),
    pytest.param({"kinds": ("TM", "TE")}, 0.0, id="TM_0m+TE_0m"),
    pytest.param({"orders": (2, -2)}, 0.0, id="HE_2m+EH_2m"),
    pytest.param({"orders": (3, -3)}, 0.4, id="HE_3m+EH_3m-rotated"),
)


@pytest.mark.parametrize(("family", "phi"), VECTOR_ORDERS)
def test_the_vector_connection_is_the_derivative_of_the_implemented_fields(family, phi):
    """Each family carries its own `C_mn`, and the two families do not connect.

    The derivation: a taper moves the radius alone, and
    every channel's field is its radial function of `r/a` times an angular pattern
    that does not depend on `a`, so `<psi^A_m|d_z psi^B_n>` is a radial integral
    times the two families' angular overlap -- `cos(2 nu theta)` for the two hybrid
    families of an order, a pointwise zero for `TM_0m` against `TE_0m` -- which
    vanishes exactly; within a family it is the scalar block's formula with the
    family's own Bessel order `p`, zeros of `J_p` and slope signs `sgn J_{p+1}(u_m)`.

    Bounds, stated first. The measurement is independent of the formula
    (:func:`implemented_connection`). Its difference step moves the radius by 1e-5,
    so the field's rounding enters as about `eps/1e-5 = 2e-11` of the connection's
    scale and the fourth-order truncation as `(1e-5)^4`, which is nothing; the
    quadrature is exact to rounding for these polynomial-times-Bessel integrands at
    400 radial nodes. Gate **1e-9** of the largest entry within each family and on
    the diagonal. Across the families the trapezoid integrates `cos(2 nu theta)`
    exactly for `2 nu < 64`, so what is left is the rounding of the sum: gate
    **1e-12**. A cross-family term of the size a hidden `a` dependence of an angular
    pattern would give -- the `(a'/a)` scale itself -- would be ten orders above it.
    Measured: 1.7e-11 within, 8.0e-14 across, on every case including the rotated
    `nu = 3` pair (the `phi` of both families turned by 0.4 rad).
    """
    profile = Linear(50e-6, 40e-6, 2e-4)
    modes = vector_block(
        (3, 3) if phi == 0.0 else (2, 2),
        radius=profile,
        gas=None,
        pressure=0.0,
        phi=phi,
        **family,
    )
    link, gram = implemented_connection(modes, profile, 1e-4)
    assert np.max(abs(gram - np.eye(len(modes)))) < 1e-12
    formula = wall_connection(modes, 1e-4, impedance="te_tm")
    scale = float(np.max(abs(formula)))
    half = len(modes) // 2
    assert scale > 0.0
    assert np.all(formula[:half, half:] == 0.0) and np.all(formula[half:, :half] == 0.0)
    for block_rows in (slice(0, half), slice(half, None)):
        assert_allclose(
            link[block_rows, block_rows],
            formula[block_rows, block_rows],
            rtol=0.0,
            atol=1e-9 * scale,
        )
    assert float(np.max(abs(link[:half, half:]))) < 1e-12 * scale
    assert float(np.max(abs(link[half:, :half]))) < 1e-12 * scale


def test_the_vector_connection_is_each_familys_own_and_one_family_is_the_scalar_one():
    """Block diagonal, real, antisymmetric; `HE_1m` alone is the scalar one.

    Bound, stated first: 4 eps of the largest entry for the single `HE_1m` family,
    because its Bessel order is zero and the family formula is then the same
    arithmetic on the same zeros, reached through another function, so only the
    rounding of a reordering separates them; exact antisymmetry because swapping the
    indices flips the denominator alone; and the `EH_1m` block is **not** the `HE_1m`
    formula transplanted -- its zeros and signs are `J_2`'s and `J_3`'s -- which the
    last assertion checks against the `p = 0` matrix of the same count.
    """
    profile = Linear(125e-6, 100e-6, 1e-2)
    modes = vector_block((3, 2), radius=profile)
    link = wall_connection(modes, 5e-3, impedance="te_tm")
    assert link.dtype == np.float64
    assert np.array_equal(link, -link.T)
    assert np.all(link[:3, 3:] == 0.0) and np.all(link[3:, :3] == 0.0)
    scalar = vector_block((3,), orders=(1,), radius=profile)
    alone = wall_connection(scalar, 5e-3)
    rounding = 4 * np.finfo(float).eps * float(np.max(np.abs(alone)))
    assert_allclose(link[:3, :3], alone, rtol=0, atol=rounding)
    assert_allclose(
        wall_connection(scalar, 5e-3, impedance="te_tm"), alone, rtol=0, atol=rounding
    )
    zeros = jn_zeros(2, 2)
    signs = np.sign(jv(3, zeros))
    rate = float(profile.derivative(np.array([5e-3]))[0]) / float(
        profile(np.array([5e-3]))[0]
    )
    expected = (
        rate
        * 2.0
        * signs[0]
        * signs[1]
        * zeros[0]
        * zeros[1]
        / (zeros[0] ** 2 - zeros[1] ** 2)
    )
    assert link[3, 4] == pytest.approx(expected, rel=1e-14)
    assert abs(link[3, 4] - link[0, 1]) > 0.1 * abs(link[0, 1])
    with pytest.raises(ModeError, match="at most two radial families"):
        wall_connection(
            vector_block((1, 1, 1), orders=(1, -1, 2), radius=profile),
            5e-3,
            impedance="te_tm",
        )


def test_the_wall_block_is_symmetric_in_the_field_amplitude_and_not_the_power_one():
    """Where the wall block's off-diagonal is right: the field amplitude.

    The question: the taper connection belongs in the
    true-power amplitude `b = W^(1/2) a`, and is the wall block's off-diagonal,
    derived in the field amplitude, wrong in the same way? No, and the reason is
    reciprocity rather than a convention. `M = k^2 I + L_N/a^2` is the matrix of a
    reciprocal operator (the Laplacian with the Robin condition) in a **real**
    orthonormal field basis, so it is complex **symmetric** there, and so is every
    function of it -- the exact one-way generator `i(sqrt(M) - k)` included, whatever
    the wall's loss. Conjugated by the diagonal `W^(1/2)` it is not. The connection is
    different because it is not a function of `M`: power conservation pins it in `b`.

    Bounds, stated first. (1) The exact square root, the independent reference that
    shares nothing with the paraxial reduction but `M`, is symmetric in the field
    amplitude to the rounding of `sqrtm` on a matrix of norm `k^2`, gated at
    `1e-13 k` (`k` about 2e6 to 8e6 1/m here; measured 1.2e-15 k), for the scalar
    block and the vector one. (2) Its `W^(1/2)` conjugate departs from symmetry by the
    weights' spread times the off-diagonal, gated above `0.1 x spread x max|G_mn|`
    (measured 0.34 1/m at 3.2 um, eight orders above the rounding). (3) What the
    conjugation would change, `(w_m - w_n)/(2w) G_mn`, is smaller than the paraxial
    reduction's own departure from the exact root, which is symmetric and of order
    `(delta beta_m + delta beta_n)/(2k)`: measured 2.2 to 2.4 times smaller at 3.2 um
    and 800 nm on both blocks -- a change below the model's own approximation, and in
    the wrong symmetry class. Measured on the scalar block at 3.2 um (800 nm):
    `|dG|` 0.172 (6.6e-4) 1/m against `max|G_mn|` 148 (9.1) 1/m and a spread of
    3.6e-3 (2.2e-4) -- over 0.2 mm, 3.4e-5 (1.3e-7) of phase; the vector block's are
    within 10 % of these.
    """
    omega = 2.0 * np.pi * C / np.array([3.2e-6, 800e-9])
    for modes, build in (
        (block(3, 50e-6, None, 0.0), wall_operator),
        (
            vector_block((2, 2), radius=50e-6, gas=None, pressure=0.0),
            vector_wall_operator,
        ),
    ):
        operator = build(modes, omega, validity_limit=2.0)
        count = len(modes)
        off = ~np.eye(count, dtype=bool)
        for index in range(omega.size):
            generator = operator.generator[:, :, index]
            carrier = complex(operator.wavenumber[index])
            weights = np.array(
                [
                    float(np.real(mode.neff(omega[index : index + 1])[0]))
                    for mode in modes
                ]
            )
            root = np.sqrt(weights)
            exact = 1j * (
                sqrtm(carrier**2 * np.eye(count) - 2j * carrier * generator)
                - carrier * np.eye(count)
            )
            assert float(np.max(abs(exact - exact.T))) < 1e-13 * abs(carrier)
            conjugated = root[:, None] * exact / root[None, :]
            spread = float(np.max(weights) - np.min(weights)) / float(np.min(weights))
            largest = float(np.max(abs(generator[off])))
            asymmetry = float(np.max(abs(conjugated - conjugated.T)))
            assert asymmetry > 0.1 * spread * largest, (asymmetry, spread, largest)
            change = float(
                np.max(
                    abs((root[:, None] * generator / root[None, :] - generator)[off])
                )
            )
            paraxial = float(np.max(abs((exact - generator)[off])))
            assert change < paraxial, (change, paraxial)

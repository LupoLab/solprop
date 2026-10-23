# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import math
from typing import Any

import numpy as np
import pytest

from solprop.constants import EPS0, C
from solprop.driver import Simulation
from solprop.exceptions import ModeError
from solprop.modes import MarcatiliMode
from solprop.modes.base import alpha, beta, beta1
from solprop.modes.protocol import Mode
from solprop.nonlinear.gnlse import (
    SpecifiedMode,
    TabulatedDispersion,
    TaylorDispersion,
    attenuation_from_db,
    kerr_susceptibility,
    problem_from,
    raman_susceptibility,
)
from solprop.output.schema import serialise
from solprop.physdata.glasses import SilicaBundle
from solprop.profiles import Linear

OMEGA_REF = 1.8e15  # rad/s, about 1030 nm
# A silica-like fibre: n_eff 1.45 at the reference, 20 fs^2/m of group-delay dispersion
# and a little third order. beta_0 and beta_1 only fix the frame.
BETAS = [1.45 * OMEGA_REF / C, 1.47 / C, 2.0e-26, -1.0e-41]


@pytest.fixture
def mode():
    return SpecifiedMode.from_taylor(BETAS, OMEGA_REF, aeff=8e-11)


def taylor_beta(omega):
    """The Taylor series of BETAS, written out."""
    delta = omega - OMEGA_REF
    return sum(b * delta**n / math.factorial(n) for n, b in enumerate(BETAS))


# ----------------------------------------------------------------------- dB/m loss


@pytest.mark.parametrize(
    ("db_per_m", "expected"),
    [(0.0, 0.0), (10.0, math.log(10.0)), (-3.0, -0.3 * math.log(10.0))],
)
def test_attenuation_from_db_is_the_power_coefficient(db_per_m, expected):
    assert attenuation_from_db(db_per_m) == pytest.approx(expected, rel=1e-15)


def test_a_loss_in_db_per_m_attenuates_the_power_by_that_many_decibels():
    # The definition, exercised the way a user states it: 3 dB/m must halve the power
    # in one metre, through the alpha the linear operator will use.
    alpha = attenuation_from_db(3.0102999566398116)
    assert math.exp(-alpha * 1.0) == pytest.approx(0.5, rel=1e-15)


def test_loss_appears_in_the_effective_index_as_an_attenuation(mode):
    lossy = SpecifiedMode.from_taylor(
        BETAS, OMEGA_REF, loss=attenuation_from_db(2.5), aeff=8e-11
    )
    omega = np.array([OMEGA_REF])
    # alpha = 2 (omega/c) Im n_eff (conventions.md section 2), inverted here.
    recovered = 2.0 * omega / C * np.imag(lossy.neff(omega))
    assert recovered[0] == pytest.approx(attenuation_from_db(2.5), rel=1e-14)
    assert np.imag(mode.neff(omega))[0] == 0.0


def test_a_callable_loss_is_evaluated_at_each_frequency():
    def loss(omega):
        return 1e-3 * omega / OMEGA_REF

    lossy = SpecifiedMode.from_taylor(BETAS, OMEGA_REF, loss=loss)
    omega = np.array([1.0e15, 2.0e15])
    assert np.allclose(lossy.alpha(omega), loss(omega), rtol=1e-15, atol=0.0)


# ------------------------------------------------------------------ the dispersion


def test_the_taylor_mode_reproduces_its_own_series(mode):
    omega = np.linspace(1.0e15, 2.6e15, 17)
    # Both sides are the same polynomial evaluated in different bases (Horner in
    # numpy.polynomial against a term-by-term sum), so only round-off separates them.
    assert np.allclose(mode.beta(omega), taylor_beta(omega), rtol=1e-13, atol=0.0)


def test_the_taylor_derivative_matches_the_differentiated_series(mode):
    omega = np.linspace(1.0e15, 2.6e15, 17)
    delta = omega - OMEGA_REF
    expected = sum(
        b * n * delta ** (n - 1) / math.factorial(n)
        for n, b in enumerate(BETAS)
        if n >= 1
    )
    assert np.allclose(
        mode.dispersion_derivative(omega), expected, rtol=1e-13, atol=0.0
    )


def test_a_tabulated_mode_agrees_with_the_taylor_mode_it_was_sampled_from(mode):
    nodes = np.linspace(1.0e15, 2.6e15, 4001)
    tabulated = SpecifiedMode.from_table(nodes, mode.beta(nodes), aeff=8e-11)
    omega = np.linspace(1.05e15, 2.55e15, 23)
    # A cubic spline on a uniform mesh of spacing h has an interpolation error of
    # order h^4 |beta''''|. Here h = 4e11 rad/s and beta'''' is zero for this cubic
    # series, so the residual is the spline solve's own round-off on numbers of size
    # beta ~ 9e6; a relative 1e-12 is generous for that.
    assert np.allclose(tabulated.beta(omega), mode.beta(omega), rtol=1e-12, atol=0.0)
    assert np.allclose(
        tabulated.dispersion_derivative(omega),
        mode.dispersion_derivative(omega),
        rtol=1e-9,
        atol=0.0,
    )


def test_a_tabulated_mode_refuses_to_extrapolate():
    nodes = np.linspace(1.0e15, 2.0e15, 8)
    tabulated = SpecifiedMode.from_table(nodes, 5.0e-9 * nodes)
    with pytest.raises(ModeError, match="outside the tabulated range"):
        tabulated.beta(np.array([2.5e15]))
    with pytest.raises(ModeError, match="outside the tabulated range"):
        tabulated.dneff_domega(np.array([0.5e15]))


def test_an_empty_request_of_a_tabulated_mode_is_allowed():
    nodes = np.linspace(1.0e15, 2.0e15, 8)
    tabulated = SpecifiedMode.from_table(nodes, 5.0e-9 * nodes)
    assert tabulated.beta(np.zeros(0)).shape == (0,)


@pytest.mark.parametrize(
    ("omega", "beta", "message"),
    [
        (np.linspace(1.0, 2.0, 3), np.zeros(3), "at least 4 points"),
        (np.array([3.0, 1.0, 2.0, 4.0]), np.zeros(4), "strictly increasing"),
    ],
)
def test_a_bad_table_is_rejected(omega, beta, message):
    with pytest.raises(ModeError, match=message):
        SpecifiedMode.from_table(omega, beta)


def test_a_table_whose_two_arrays_disagree_is_rejected():
    # Through the callable directly: `from_table` annotates both arrays with the same
    # axis name, so the hook rejects the mismatch before the library's own check.
    # TabulatedDispersion gives them independent names, which is what keeps that check
    # reachable in production, where no hook runs.
    with pytest.raises(ModeError, match="shapes"):
        TabulatedDispersion(np.linspace(1.0, 2.0, 8), np.zeros(7))


def test_the_two_dispersion_callables_serialise_reproducibly():
    """Two modes built the same way must record the same configuration.

    A closure is recorded by `serialise` as its `repr`, which carries a memory address,
    so a run whose metadata held one could never be resumed from another process: its
    configuration hash would change every time. The callables are dataclasses for that
    reason, and this is the property that buys.
    """
    build = lambda betas: SpecifiedMode.from_taylor(  # noqa: E731
        betas, omega_ref=OMEGA_REF, aeff=8e-11
    )
    one, again = build(BETAS), build(BETAS)
    assert json.dumps(serialise(one), sort_keys=True) == json.dumps(
        serialise(again), sort_keys=True
    )
    # A list and an array of the same coefficients must hash alike, and a changed
    # coefficient must not.
    assert json.dumps(
        serialise(build(np.asarray(BETAS))), sort_keys=True
    ) == json.dumps(serialise(one), sort_keys=True)
    assert json.dumps(serialise(build([*BETAS[:2], 3.0e-26])), sort_keys=True) != (
        json.dumps(serialise(one), sort_keys=True)
    )


def _gnlse_key(fibre):
    """Return the resume key of a millimetre run through the given fibre."""
    return (
        Simulation.from_gnlse(
            0.1,
            1e-3,
            fibre=fibre,
            lambda0=835e-9,
            wavelength_limits=(600e-9, 1300e-9),
            trange=2e-12,
            fwhm=200e-15,
            energy=1e-12,
            n_save=3,
            noise=None,
            backend="numpy_ref",
        )
        .metadata()
        .configuration_hash()
    )


def test_a_fibre_that_moves_along_z_records_the_same_configuration_twice():
    """Two identical moving fibres must produce the same resume key.

    The same property as the frequency-only case above, one level up: a declared
    path and a mode-derived one both enter :meth:`RunMetadata.configuration_hash`,
    and `_ModeArea` is a frozen dataclass rather than a closure precisely so that
    the key does not carry a memory address and change from one process to the
    next. The two fibres must also not hash **alike**, or the key would not be
    telling the runs apart.
    """

    def declared():
        return SpecifiedMode.from_taylor_path(
            [BETAS, [*BETAS[:1], 1.5 / C, *BETAS[2:]]],
            [0.0, 1e-3],
            omega_ref=OMEGA_REF,
            aeff=[8e-11, 6e-11],
        )

    def wrapped():
        return SpecifiedMode.from_mode(
            MarcatiliMode(Linear(15e-6, 12e-6, 1e-3), cladding=1.45)
        )

    assert _gnlse_key(declared()) == _gnlse_key(declared())
    assert _gnlse_key(wrapped()) == _gnlse_key(wrapped())
    assert _gnlse_key(declared()) != _gnlse_key(wrapped())


def test_a_tabulated_dispersion_serialises_its_table():
    nodes = np.linspace(1.0e15, 2.0e15, 8)
    # Loosely typed: `serialise` promises JSON, and a test walks into it.
    record: Any = serialise(TabulatedDispersion(nodes, 5.0e-9 * nodes))
    assert record["__class__"].endswith("TabulatedDispersion")
    # The spline is derived in __post_init__ and declared init=False, so it is not part
    # of what a run records; the table it was built from is.
    assert set(record) == {"__class__", "omega", "beta", "derivative"}
    assert record["omega"]["__ndarray__"] == nodes.tolist()


@pytest.mark.parametrize(
    ("kind", "arguments"),
    [(TaylorDispersion, ([1.0], 1.0e15, -1)), (TabulatedDispersion, None)],
)
def test_a_negative_derivative_order_is_rejected(kind, arguments):
    nodes = np.linspace(1.0e15, 2.0e15, 8)
    with pytest.raises(ModeError, match="non-negative order"):
        kind(*(arguments if arguments is not None else (nodes, 5.0e-9 * nodes, -1)))


def test_a_taylor_series_with_a_non_finite_coefficient_is_rejected():
    with pytest.raises(ModeError, match="must be finite"):
        TaylorDispersion([0.0, np.inf], 1.0e15)


def test_the_second_derivative_of_each_callable_is_the_group_velocity_dispersion():
    """Both callables differentiate to any order, exactly in their own basis."""
    omega = np.linspace(1.2e15, 2.4e15, 9)
    taylor = TaylorDispersion(BETAS, OMEGA_REF, 2)
    expected = BETAS[2] + BETAS[3] * (omega - OMEGA_REF)
    assert np.allclose(taylor(omega), expected, rtol=1e-12, atol=0.0)
    nodes = np.linspace(1.0e15, 2.6e15, 4001)
    tabulated = TabulatedDispersion(nodes, TaylorDispersion(BETAS, OMEGA_REF)(nodes), 2)
    # A not-a-knot spline reproduces a cubic exactly, so what is left is round-off
    # amplified by the second difference: eps |beta| / h^2 = 2.2e-16 x 1.26e7 / (4e11)^2
    # = 1.8e-32, which against a beta_2 of 2e-26 is 9e-7 relative. Measured 7.5e-7; the
    # bound keeps an order of margin.
    assert np.allclose(tabulated(omega), expected, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"betas": [], "omega_ref": 1.0e15}, "at least the zeroth-order"),
        ({"betas": [1.0], "omega_ref": 0.0}, "omega_ref must be positive"),
        ({"betas": [1.0], "omega_ref": 1.0e15, "aeff": 0.0}, "effective_area"),
    ],
)
def test_a_bad_taylor_specification_is_rejected(kwargs, message):
    with pytest.raises(ModeError, match=message):
        SpecifiedMode.from_taylor(**kwargs)


# --------------------------------------------------------------- the mode protocol


def test_the_mode_satisfies_the_protocol(mode):
    assert isinstance(mode, Mode)
    assert (mode.z_constant, mode.scale_invariant, mode.azimuthal_order) == (
        True,
        True,
        0,
    )


def test_the_effective_index_inverts_beta_and_alpha(mode):
    omega = np.linspace(1.0e15, 2.6e15, 11)
    index = mode.neff(omega)
    # conventions.md section 2: beta = (omega/c) Re n_eff, alpha = 2(omega/c) Im n_eff.
    assert np.allclose(omega / C * index.real, mode.beta(omega), rtol=1e-14, atol=0.0)


def test_neff_minus_one_agrees_with_the_subtraction_it_avoids(mode):
    omega = np.linspace(1.0e15, 2.6e15, 11)
    # Here n_eff is 1.45, far from one, so the two forms may be compared directly; the
    # point of the direct form is the gas-filled case, where they could not be.
    assert np.allclose(
        mode.neff_minus_one(omega), mode.neff(omega) - 1.0, rtol=1e-14, atol=0.0
    )


@pytest.mark.parametrize("method", ["neff", "neff_minus_one", "dneff_domega"])
def test_the_effective_index_is_undefined_at_zero_frequency(mode, method):
    with pytest.raises(ModeError, match="no effective index at or below zero"):
        getattr(mode, method)(np.array([0.0, 1.0e15]))


def test_the_analytic_derivative_matches_finite_differences(mode):
    # The ADR-006 gate: the analytic dn/dw must agree with a central difference, and
    # the disagreement must fall as the square of the step until round-off takes over.
    omega = OMEGA_REF
    analytic = mode.dneff_domega(np.array([omega]))[0].real
    errors = []
    for relative in (9e-3, 3e-3, 1e-3):
        h = relative * omega
        differenced = (
            mode.neff(np.array([omega + h]))[0].real
            - mode.neff(np.array([omega - h]))[0].real
        ) / (2.0 * h)
        errors.append(abs(differenced / analytic - 1.0))
    # Central differences are second order, so each threefold reduction of the step
    # must cut the error by about nine. Round-off on n_eff ~ 1.45 divided by
    # 2h ~ 4e12 contributes about 1e-28 absolute, far below the smallest truncation
    # error reached here.
    assert errors[0] / errors[1] == pytest.approx(9.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(9.0, rel=0.05)
    # The residual at the smallest step is the central difference's own h^2
    # truncation error, measured at 6.5e-7; it is the reference that is inaccurate,
    # not the analytic derivative, which is exactly the point of ADR-006.
    assert errors[2] < 1e-6


def test_the_transverse_profile_is_the_disc_of_the_declared_effective_area(mode):
    radius = mode.radius()
    assert radius == pytest.approx(math.sqrt(mode.aeff() / math.pi), rel=1e-15)
    kind, lower, upper = mode.dimlimits()
    assert kind == "polar"
    assert lower == (0.0, 0.0)
    assert upper == pytest.approx((radius, 2.0 * math.pi))

    # A_eff = (int |e|^2 dA)^2 / int |e|^4 dA over the disc, by the midpoint rule in r
    # with the polar Jacobian; the profile is flat, so the rule is exact in r and the
    # only error is the radial cell that straddles the edge.
    r_edges = np.linspace(0.0, 2.0 * radius, 20001)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]
    points = np.stack([r, np.zeros_like(r)], axis=1)
    profile = mode.field(points)[:, 0]
    weight = 2.0 * math.pi * r * dr
    area = np.sum(profile**2 * weight) ** 2 / np.sum(profile**4 * weight)
    assert area == pytest.approx(mode.aeff(), rel=1e-4)


def test_the_profile_is_polarised_along_x(mode):
    points = np.array([[0.0, 0.0], [10.0 * mode.radius(), 1.0]])
    assert np.array_equal(mode.field(points), np.array([[1.0, 0.0], [0.0, 0.0]]))


# ------------------------------------------------------------------- the mapping


@pytest.mark.parametrize("gamma", [1e-3, 0.1, 37.0])
@pytest.mark.parametrize("aeff", [1.0, 8e-11])
def test_gamma_makes_a_round_trip_through_n2(gamma, aeff):
    problem = problem_from(BETAS, omega_ref=OMEGA_REF, gamma=gamma, aeff=aeff)
    assert problem.gamma == pytest.approx(gamma, rel=1e-14)
    # The definition the round trip rests on: gamma = n2 omega0 / (c A_eff).
    assert problem.n2 * OMEGA_REF / (C * aeff) == pytest.approx(gamma, rel=1e-14)
    from_index = problem_from(BETAS, omega_ref=OMEGA_REF, n2=problem.n2, aeff=aeff)
    assert from_index.gamma == pytest.approx(gamma, rel=1e-14)


def test_the_susceptibility_is_luna_s_four_thirds_mapping():
    problem = problem_from(BETAS, omega_ref=OMEGA_REF, gamma=0.1)
    # chi3 = (4/3) n2 eps0 c with the linear index taken as one (ADR-026).
    assert problem.chi3 == pytest.approx(4.0 / 3.0 * problem.n2 * EPS0 * C, rel=1e-15)
    assert problem.kerr.chi3 == problem.chi3


def test_the_whole_of_gamma_acts_as_kerr_when_raman_is_off():
    # With raman = false the (1 - fr) factor must go too; keeping it would propagate
    # 0.82 of the nonlinearity the user asked for.
    without = problem_from(BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=False)
    assert without.raman_fraction == 0.0
    assert without.chi3 == pytest.approx(4.0 / 3.0 * without.n2 * EPS0 * C, rel=1e-15)


def test_raman_takes_its_share_of_gamma_and_leaves_the_rest_to_the_kerr_term():
    problem = problem_from(
        BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=True, raman_fraction=0.18
    )
    assert problem.raman_fraction == 0.18
    assert problem.raman is not None
    assert problem.raman.fraction == 0.18
    # An envelope GNLSE grid carries no third harmonic, on either response, and the
    # translation layer checks that the two agree.
    assert problem.raman.thg is False
    assert problem.responses == (problem.kerr, problem.raman)
    assert problem.chi3 == pytest.approx(kerr_susceptibility(problem.n2, 0.18))


def test_the_two_susceptibilities_add_up_to_the_whole_nonlinearity():
    """The identity the mapping is built on, and the reason for the factor of two.

    A Raman term of integral ``X`` acts like a Kerr term of ``(2/3) X`` in the slowly
    varying limit, because the envelope drive carries a half where the envelope cubic
    carries three quarters. So the share the Raman response takes is
    ``(2/3) f_R raman_susceptibility(n2)`` and it is exactly what
    ``kerr_susceptibility(n2, f_R)`` leaves behind.
    """
    n2, fraction = 2.6e-20, 0.18
    total = kerr_susceptibility(n2)
    assert raman_susceptibility(n2) == pytest.approx(2.0 * n2 * EPS0 * C, rel=1e-15)
    assert raman_susceptibility(n2) == pytest.approx(1.5 * total, rel=1e-15)
    share = (2.0 / 3.0) * fraction * raman_susceptibility(n2)
    assert share + kerr_susceptibility(n2, fraction) == pytest.approx(total, rel=1e-15)


def test_the_raman_response_integrates_to_its_share(mode):
    """The response the mapping builds carries the integral the identity asks for."""
    problem = problem_from(
        BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=True, raman_fraction=0.18
    )
    dt = 1e-16
    delays = np.arange(1 << 20) * dt
    assert problem.raman is not None
    integral = float(np.trapezoid(problem.raman.impulse(delays, 1.0), dx=dt))
    expected = 0.18 * raman_susceptibility(problem.n2)
    # The shape is normalised analytically and the check is the trapezoid rule on a
    # 105 ps window at 0.1 fs; measured 4.0e-6 relative, which is the rule's own error
    # on an integrand that oscillates at 14 THz.
    assert integral == pytest.approx(expected, rel=1e-5)


def test_raman_with_a_zero_fraction_is_refused():
    # A declared response that carries none of the nonlinearity would pay for a
    # convolution at every stage and change nothing; RamanResponse refuses the fraction
    # in any case.
    with pytest.raises(ModeError, match="carries none of the nonlinearity"):
        problem_from(
            BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=True, raman_fraction=0.0
        )


def test_the_default_fraction_is_silica_s_with_raman_on_and_none_with_it_off():
    on = problem_from(BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=True)
    off = problem_from(BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=False)
    # Silica's is optmat's bundle's (SEP 3, D4): 0.2188; the inline 0.18 is gone.
    assert on.raman_fraction == SilicaBundle.resolve().raman_fraction
    assert f"{on.raman_fraction:.4f}" == "0.2188"
    assert on.raman is not None
    assert on.raman.fraction == on.raman_fraction
    assert off.raman_fraction == 0.0
    assert off.chi3 == pytest.approx(kerr_susceptibility(off.n2), rel=1e-15)


def test_a_given_fraction_is_held_back_with_raman_off():
    """Raman off with an explicit fraction keeps the Kerr term of Raman on.

    So a pair of runs that gives the same fraction and switches only ``raman`` differs
    by the delayed response alone, which is Luna's ``prop_gnlse`` convention.
    """
    off = problem_from(
        BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=False, raman_fraction=0.18
    )
    on = problem_from(
        BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=True, raman_fraction=0.18
    )
    assert off.raman is None
    assert off.responses == (off.kerr,)
    assert off.raman_fraction == 0.18
    assert off.chi3 == on.chi3
    assert off.chi3 == pytest.approx(kerr_susceptibility(off.n2, 0.18), rel=1e-15)


def test_a_given_zero_fraction_with_raman_off_is_the_whole_of_gamma():
    zero = problem_from(
        BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=False, raman_fraction=0.0
    )
    default = problem_from(BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=False)
    assert zero.chi3 == default.chi3
    assert zero.raman_fraction == 0.0


def test_raman_off_declares_no_delayed_response():
    problem = problem_from(BETAS, omega_ref=OMEGA_REF, gamma=0.1, raman=False)
    assert problem.raman is None
    assert problem.responses == (problem.kerr,)


def test_the_susceptibility_split_reserves_the_raman_fraction():
    whole = kerr_susceptibility(1e-20)
    assert whole == pytest.approx(4.0 / 3.0 * 1e-20 * EPS0 * C, rel=1e-15)
    assert kerr_susceptibility(1e-20, 0.18) / whole == pytest.approx(0.82, rel=1e-14)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_the_susceptibility_split_rejects_a_bad_fraction(fraction):
    with pytest.raises(ValueError, match="raman_fraction must lie"):
        kerr_susceptibility(1e-20, fraction)


def test_the_mapping_records_what_it_was_given():
    problem = problem_from(
        BETAS,
        omega_ref=OMEGA_REF,
        gamma=0.1,
        aeff=8e-11,
        loss_db_per_m=2.5,
        shock=False,
    )
    assert problem.omega0 == OMEGA_REF
    assert problem.aeff == 8e-11
    assert problem.shock is False
    assert problem.mode.aeff() == 8e-11
    assert isinstance(problem.mode, SpecifiedMode)
    assert problem.mode.alpha(np.array([OMEGA_REF]))[0] == pytest.approx(
        attenuation_from_db(2.5), rel=1e-15
    )
    # An envelope GNLSE grid carries no third harmonic.
    assert problem.kerr.thg is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one of gamma and n2"),
        ({"gamma": 0.1, "n2": 1e-20}, "exactly one of gamma and n2"),
        ({"gamma": 0.1, "raman_fraction": 1.5}, r"raman_fraction must lie in \[0, 1\]"),
        (
            {"gamma": 0.1, "raman": True, "raman_fraction": -0.1},
            r"raman_fraction must lie in \[0, 1\]",
        ),
        ({"gamma": 0.1, "aeff": 0.0}, "aeff must be positive"),
    ],
)
def test_a_bad_mapping_request_is_rejected(kwargs, message):
    with pytest.raises(ModeError, match=message):
        problem_from(BETAS, omega_ref=OMEGA_REF, **kwargs)


def test_from_mode_preserves_linear_values_and_analytic_derivative():
    source = MarcatiliMode(15e-6, cladding=1.45)
    omega = np.linspace(1e15, 4e15, 20)
    mode = SpecifiedMode.from_mode(source)
    assert mode.aeff() == source.aeff()
    # Structural: the specified mode delegates to its source, so the values are the
    # source's own, exactly.
    np.testing.assert_array_equal(mode.beta(omega), beta(source, omega))
    np.testing.assert_array_equal(
        mode.dispersion_derivative(omega), beta1(source, omega)
    )
    np.testing.assert_array_equal(mode.alpha(omega), alpha(source, omega))
    assert SpecifiedMode.from_mode(source, aeff=1e-10).aeff() == 1e-10
    with pytest.raises(ModeError, match="effective_area must be positive"):
        SpecifiedMode.from_mode(source, aeff=0.0)
    # A source whose bore tapers now hands over both paths rather than being refused; a
    # constant area beside a moving dispersion is what is refused instead.
    moving = MarcatiliMode(Linear(15e-6, 10e-6, 1.0), cladding=1.45)
    carried = SpecifiedMode.from_mode(moving)
    assert not carried.z_constant
    np.testing.assert_array_equal(carried.beta(omega, 0.4), beta(moving, omega, 0.4))
    assert carried.aeff(0.4) == moving.aeff(0.4)
    with pytest.raises(ModeError, match="needs an effective-area"):
        SpecifiedMode.from_mode(moving, aeff=1e-10)
    # And a source whose *fill* moves, which the bore case does not cover: a fixed
    # bore on a pressure gradient declares its density path instead of a geometry
    # one, so the area comes from the `FromCallable(_ModeArea(...))` branch and is
    # a path whose value happens not to change.
    filled = MarcatiliMode(15e-6, "Ar", Linear(1.0, 2.0, 1.0))
    gradient = SpecifiedMode.from_mode(filled)
    # Inside argon's fitted band (110-1000 nm), so the source's own range warning
    # is not what this case is about.
    band = np.linspace(1.9e15, 2.6e15, 5)
    assert not gradient.z_constant
    assert not gradient.dispersion_is_constant
    np.testing.assert_array_equal(gradient.beta(band, 0.4), beta(filled, band, 0.4))
    assert float(gradient.beta(band, 1.0)[0]) != float(gradient.beta(band, 0.0)[0])
    assert gradient.aeff(1.0) == gradient.aeff(0.0) == filled.aeff()

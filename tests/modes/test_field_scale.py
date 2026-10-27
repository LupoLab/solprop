# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The field-scale radial rules of SEP 19.

For one HE_11 mode in a uniform medium every node sees the axial waveform scaled by
``eta = J_0(u xi)``, so the radial projection is an integral over the field scale, and
the Gauss and Gauss-Radau rules for that measure integrate the linear and Kerr source
exactly from two (Radau: three) nodes. The checks here are the rule's own moments
against independent adaptive integrals, the guards that keep it to the one profile it
is exact for, the area-weight convention that lets a node sit on the axis, and the
two identities SEP 19 section 4.3 states: source exactness against a dense rule, and
the one-node member as the effective-area model.
"""

import dataclasses
import math
import warnings
from contextlib import closing

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, j1, jn_zeros

from rate_oracles import adk_oracle
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import MaterialRangeWarning, ModeError, OutputError
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.modes.quadrature import (
    BASIS_NORMALISATION,
    RadialRule,
    field_scale_rule,
    mode_matrix,
    quadrature_rule,
)
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Gas
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble

U01 = float(jn_zeros(0, 1)[0])
GAUSS_ORDERS = (2, 3, 5, 8, 16, 32)
RADAU_ORDERS = (4, 5, 8, 16, 32)
# SEP 19 section 7's target, which the rule is certified to on construction against
# its refined measure. Measured against these adaptive integrals (themselves good to
# 2e-14): residuals of about 3e-14 at every order, the largest 1.1e-13 over Gauss and
# Radau orders 2 to 64.
MOMENT_TOLERANCE = 1e-12


def moment(k):
    """``M_k = int_0^1 J_0(u xi)^(k+1) xi d xi``, by adaptive quadrature."""
    value, error = quad(
        lambda xi: xi * j0(U01 * xi) ** (k + 1),
        0.0,
        1.0,
        epsabs=0.0,
        epsrel=2e-14,
        limit=400,
    )
    assert error < 0.05 * MOMENT_TOLERANCE * abs(value)
    return value


@pytest.mark.parametrize("n", GAUSS_ORDERS)
def test_the_gauss_rule_integrates_its_degree_against_independent_moments(n):
    eta, weights = field_scale_rule(n)
    for k in range(2 * n):
        exact = moment(k)
        assert abs(np.sum(weights * eta**k) - exact) <= MOMENT_TOLERANCE * exact, k


@pytest.mark.parametrize("n", RADAU_ORDERS)
def test_the_radau_rule_integrates_its_degree_and_keeps_its_axis_node(n):
    eta, weights = field_scale_rule(n, radau=True)
    assert eta[0] == 1.0
    assert np.all(eta[1:] < 1.0)
    for k in range(2 * n - 1):
        exact = moment(k)
        assert abs(np.sum(weights * eta**k) - exact) <= MOMENT_TOLERANCE * exact, k


def test_the_low_moments_are_the_closed_forms():
    """``M_0 = J_1(u)/u`` and ``M_1 = J_1(u)^2 / 2``, with ``J_0(u) = 0``."""
    eta, weights = field_scale_rule(8)
    assert math.isclose(np.sum(weights), j1(U01) / U01, rel_tol=1e-13)
    assert math.isclose(np.sum(weights * eta), j1(U01) ** 2 / 2, rel_tol=1e-13)


def test_one_node_is_the_effective_area_rule_not_the_one_point_gauss_rule():
    """It matches moments 1 and 3, so the linear and Kerr terms, at eta_e = 0.752."""
    eta, weights = field_scale_rule(1)
    first, third = moment(1), moment(3)
    assert math.isclose(eta[0], math.sqrt(third / first), rel_tol=1e-13)
    assert math.isclose(weights[0] * eta[0], first, rel_tol=1e-13)
    assert math.isclose(weights[0] * eta[0] ** 3, third, rel_tol=1e-13)
    # The one-point Gauss rule would sit at M_1 / M_0 = 0.624 instead.
    assert abs(eta[0] - 0.75202157) < 1e-8


@pytest.mark.parametrize("radau", [False, True])
def test_the_rules_are_positive_ordered_and_deterministic(radau):
    first = field_scale_rule(8, radau=radau)
    field_scale_rule.cache_clear()
    second = field_scale_rule(8, radau=radau)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    eta, weights = second
    assert np.all(weights > 0.0)
    assert np.all(np.diff(eta) < 0.0)
    assert not eta.flags.writeable


# ------------------------------------------------------------------------ the guards


@pytest.mark.parametrize(
    ("request_", "message"),
    [
        ({"kind": "cartesian", "full": True}, "radial-only on a polar core"),
        ({"kind": "slab", "full": False}, "radial-only on a polar core"),
        ({"kind": "polar", "full": True}, "radial-only on a polar core"),
        ({"kind": "polar", "full": False, "breaks": (0.5,)}, "without panels"),
    ],
)
def test_a_field_scale_rule_is_radial_only_on_a_whole_polar_core(request_, message):
    with pytest.raises(ModeError, match=message):
        quadrature_rule(n_r=8, radial_rule="field_scale", **request_)


@pytest.mark.parametrize(
    ("radial_rule", "n_r", "message"),
    [
        ("field_scale", 0, "needs n_r >= 1"),
        ("field_scale_radau", 3, "needs n_r >= 4"),
        ("gauss", 1, "needs n_r >= 2"),
        ("gauss_kronrod", 1, "needs n_r >= 2"),
        ("gauss", 2.0, "must be an integer"),
        ("gauss", True, "must be an integer"),
        ("simpson", 8, "radial_rule must be one of"),
    ],
)
def test_counts_and_names_are_refused_before_any_array_is_built(
    radial_rule, n_r, message
):
    """Through the unwrapped function: under the test hook the annotations refuse.

    The guard behind them is what production runs, so it is exercised directly.
    """
    unwrapped = getattr(quadrature_rule, "__wrapped__", quadrature_rule)
    with pytest.raises(ModeError, match=message):
        unwrapped("polar", full=False, n_r=n_r, radial_rule=radial_rule)


class _OtherProfile(MarcatiliMode):
    """An HE_11-labelled mode whose radial profile is not Marcatili's."""

    def field(self, points, z=0.0):
        r = np.asarray(points, dtype=float)[:, 0]
        profile = np.exp(-((r / self.radius) ** 2))
        return np.stack([profile, np.zeros_like(profile)], axis=-1)


@pytest.mark.parametrize(
    "modes",
    [
        [MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2)],
        [MarcatiliMode(125e-6, "He", 1.0, m=2)],
        [_OtherProfile(125e-6, "He", 1.0)],
    ],
    ids=["two modes", "HE12", "another profile"],
)
def test_only_one_marcatili_he11_profile_may_use_the_field_scale(modes):
    with pytest.raises(ModeError, match="exactly one Marcatili HE_11 mode"):
        transverse_quadrature(modes, n_r=8, radial_rule="field_scale")
    # A rule built from the geometry alone cannot reach a transform either.
    rule = quadrature_rule("polar", full=False, n_r=8, radial_rule="field_scale")
    with pytest.raises(ModeError, match="exactly one Marcatili HE_11 mode"):
        mode_matrix(modes, rule, components=("x",))


def test_a_rule_carries_exactly_one_weight_representation():
    rule = quadrature_rule("polar", full=False, n_r=8, radial_rule="field_scale")
    with pytest.raises(ModeError, match="exactly one pair"):
        dataclasses.replace(rule, w_r=rule.area_r, w_r_coarse=rule.area_r)
    gauss = quadrature_rule("polar", full=False, n_r=8)
    with pytest.raises(ModeError, match="exactly one pair"):
        dataclasses.replace(gauss, w_r=None, w_r_coarse=None)
    with pytest.raises(ModeError, match="belong to the field-scale rules"):
        dataclasses.replace(rule, radial_rule="gauss")


# ------------------------------------------------- area weights and source exactness


def test_the_radau_axis_node_has_a_finite_area_weight():
    rule = quadrature_rule("polar", full=False, n_r=5, radial_rule="field_scale_radau")
    assert rule.xi_r[0] == 0.0
    assert rule.area_r is not None and np.all(np.isfinite(rule.area_r))
    assert rule.w_r is None and not rule.has_embedded_rule


def test_area_weights_integrate_the_source_and_not_the_area():
    """Do not renormalise them: 2 sum q is 0.959 at five nodes, the area 1."""
    rule = quadrature_rule("polar", full=False, n_r=5, radial_rule="field_scale")
    assert rule.area_r is not None
    assert abs(2 * np.sum(rule.area_r) - 0.95943948) < 1e-7
    radius = 125e-6
    weights = rule.weights(("polar", (0.0, 0.0), (radius, 2 * math.pi)))
    profile = j0(U01 * rule.xi_r)
    for k in (1, 3):  # the linear and Kerr source moments
        exact = 2 * math.pi * radius**2 * moment(k)
        assert math.isclose(np.sum(weights * profile ** (k + 1)), exact, rel_tol=1e-13)


@pytest.mark.parametrize(
    ("radial_rule", "n_r"),
    [
        ("field_scale", 2),
        ("field_scale", 4),
        ("field_scale", 8),
        ("field_scale_radau", 4),
        ("field_scale_radau", 8),
    ],
)
@pytest.mark.parametrize("phi", [0.0, math.pi / 5])
def test_the_linear_and_kerr_sources_are_exact(radial_rule, n_r, phi):
    """Against a 64-node Gauss rule, for a scalar and a two-component field.

    The projected cubic of a mode-one field is ``int e^4 dA / N``: exact for these
    rules by their polynomial degree in eta, for x and for a rotated HE_11 whose two
    components each scale with eta.
    """
    mode = MarcatiliMode(125e-6, "He", 1.0, phi=phi)
    components = ("x",) if phi == 0.0 else ("x", "y")
    dense = transverse_quadrature([mode], n_r=64, radial_rule="gauss")
    rule = transverse_quadrature([mode], n_r=n_r, radial_rule=radial_rule)

    def projected(quadrature, power):
        matrices = mode_matrix([mode], quadrature, components=components)
        field = matrices.synthesis[:, 0].reshape(-1, len(components))
        magnitude = np.sum(field**2, axis=1, keepdims=True)
        source = (magnitude ** ((power - 1) / 2) * field).reshape(-1)
        return float(matrices.projection[0] @ source)

    for power in (1, 3):
        want = projected(dense, power)
        # The rule's certified moment tolerance: its measured residual is 3e-14.
        assert abs(projected(rule, power) - want) <= MOMENT_TOLERANCE * abs(want), power


def test_the_projection_is_still_the_dual_of_the_synthesis():
    """The linear moment is exact, so ``projection @ synthesis`` is the identity."""
    mode = MarcatiliMode(125e-6, "He", 1.0)
    cases: tuple[tuple[RadialRule, int], ...] = (
        ("field_scale", 2),
        ("field_scale_radau", 4),
    )
    for radial_rule, n_r in cases:
        rule = transverse_quadrature([mode], n_r=n_r, radial_rule=radial_rule)
        matrices = mode_matrix([mode], rule, components=("x",))
        assert math.isclose(
            (matrices.projection @ matrices.synthesis).item(), 1.0, rel_tol=1e-13
        )
    assert BASIS_NORMALISATION > 0


# --------------------------------------------------------------- short propagations

LENGTH = 0.02
STEPS = 10
ENERGY = 1e-4  # J: enough for appreciable argon ionisation over ten steps


@pytest.fixture(scope="module")
def grid():
    return FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 3e-6), 100e-15)


def run(grid, modes, responses, energy, **options):
    """Ten fixed steps of a short argon capillary; return the final field."""
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=energy).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = assemble(
            grid,
            modes,
            responses,
            start,
            controller=PIController(rtol=1e-10, atol=1e-16),
            save_planes=[LENGTH],
            dz_max=LENGTH / STEPS,
            dz_init=LENGTH / STEPS,
            schedule=FixedSchedule.uniform(
                0.0, LENGTH, LENGTH / STEPS, save_planes=[LENGTH]
            ),
            **options,
        )
    with closing(NumpyRefBackend(problem, grid=grid)) as backend:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        return backend.observe().require_field(), problem


def peak_gap(got, want):
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@pytest.fixture(scope="module")
def argon():
    return MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45, loss=False)


@pytest.mark.parametrize(
    ("radial_rule", "n_r"),
    [("field_scale", 2), ("field_scale", 8), ("field_scale_radau", 4)],
)
def test_a_kerr_run_matches_the_default_rule(grid, argon, radial_rule, n_r):
    """The Kerr source is exact on both rules, so ten steps agree to rounding."""
    kerr = [KerrResponse(chi3=2.2e-52, thg=True)]
    rule = transverse_quadrature([argon], n_r=n_r, radial_rule=radial_rule)
    default = transverse_quadrature([argon], n_r=32, radial_rule="gauss_kronrod")
    got, _ = run(grid, [argon], kerr, ENERGY, quadrature=rule)
    want, _ = run(grid, [argon], kerr, ENERGY, quadrature=default)
    # Measured 1.3e-16 for every rule here: the two rules project the same cubic to
    # rounding, so ten fixed steps agree to rounding. The bound is the moment
    # tolerance, well inside SEP 19 section 7's preliminary 1e-10.
    assert peak_gap(got, want) < 1e-12


@pytest.mark.parametrize("plasma", [False, True])
def test_one_field_scale_node_is_the_mode_averaged_model(grid, argon, plasma):
    """SEP 19 section 4.3: the same local response and the same projected source.

    The one-node rule's field is the effective-area field and its physical weight is
    A_eff, so it reproduces the mode-averaged route through the resolved path -- for
    the plasma as well as the Kerr term. Before the transform kind was stored, a
    one-node quadrature was read as mode-averaged and converted twice.
    """
    responses: list[KerrResponse | PlasmaResponse] = [
        KerrResponse(chi3=2.2e-52, thg=True)
    ]
    if plasma:
        table = build_rate_table(adk_oracle(Gas.Ar), (1e9, 2e11))
        responses.append(PlasmaResponse(Gas.Ar, table))
    one = transverse_quadrature([argon], n_r=1, radial_rule="field_scale")
    resolved, problem = run(grid, [argon], responses, ENERGY, quadrature=one)
    averaged, reference = run(grid, argon, responses, ENERGY)
    assert problem.transform.kind == "modal"
    assert reference.transform.kind == "mode_averaged"
    # Measured 1.7e-16 (Kerr) and 4.5e-16 (Kerr and plasma; the plasma moves the
    # field by 1.4e-4 at this energy, and the mode-averaged answer differs from a
    # resolved 33-node run by 7e-3): the routes order the same products differently,
    # so the gap is rounding through ten fixed steps.
    assert peak_gap(resolved, averaged) < 1e-12


# ------------------------------------------------------------ provenance and resume


def capillary_simulation(radial_rule, n_r):
    """A short helium capillary on a coarse grid; nothing is propagated."""
    from solprop.driver import Simulation  # noqa: PLC0415 - the driver layer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(
            125e-6,
            0.01,
            "He",
            1.0,
            lambda0=800e-9,
            fwhm=10e-15,
            energy=1e-6,
            modes=1,
            trange=100e-15,
            wavelength_limits=(300e-9, 2e-6),
            backend="numpy_ref",
            n_r=n_r,
            radial_rule=radial_rule,
            noise=None,
        )


def test_the_run_records_its_rule_and_a_changed_rule_changes_the_resume_key():
    from solprop.output.schema import RunMetadata  # noqa: PLC0415 - the output layer

    field_scale = capillary_simulation("field_scale_radau", 5).metadata()
    default = capillary_simulation("gauss_kronrod", 32).metadata()
    record = field_scale.quadrature
    assert record is not None
    assert record["radial_rule"] == "field_scale_radau"
    assert (record["requested_n_r"], record["n_nodes"]) == (5, 5)
    assert record["effective_area_node"] is False
    assert record["embedded_rule"] is False
    assert np.asarray(record["nodes"])[0, 0] == 0.0  # the Radau axis node
    assert default.quadrature is not None
    assert (default.quadrature["requested_n_r"], default.quadrature["n_nodes"]) == (
        32,
        33,
    )
    assert field_scale.configuration_hash() != default.configuration_hash()
    restored = RunMetadata.from_json(field_scale.to_json())
    assert restored.configuration_hash() == field_scale.configuration_hash()
    assert restored.quadrature is not None
    assert restored.quadrature["radial_rule"] == "field_scale_radau"


def test_the_one_node_rule_is_named_as_the_effective_area_special_case():
    metadata = capillary_simulation("field_scale", 1).metadata()
    assert metadata.simulation_kind == "multimode"
    assert metadata.quadrature is not None
    assert metadata.quadrature["effective_area_node"] is True


def test_a_record_older_than_schema_17_carries_no_quadrature():
    """Missing means "not recorded", never a field-scale rule."""
    metadata = capillary_simulation("gauss_kronrod", 32).metadata()
    legacy = dataclasses.replace(metadata, schema_version=16, quadrature=None)
    assert "quadrature" not in legacy.hashed()
    with pytest.raises(OutputError, match="requires output schema 17"):
        dataclasses.replace(metadata, schema_version=16)


# ------------------------------------------------------------ the rule's own guards


def test_a_rule_needs_its_minimum_order():
    with pytest.raises(ModeError, match="needs more nodes"):
        field_scale_rule(0)
    with pytest.raises(ModeError, match="Radau rule needs more nodes"):
        field_scale_rule(1, radau=True)


def test_a_recurrence_that_never_settles_is_refused(monkeypatch):
    from solprop.modes import quadrature  # noqa: PLC0415 - patched module

    # Starting with a coarse measure exercises a failed comparison, not merely
    # the absence of a second measure to compare.
    monkeypatch.setattr(quadrature, "_MEASURE_COUNTS", (8, 16))
    with pytest.raises(ModeError, match="did not settle"):
        quadrature._field_scale_recurrence(4)


def test_the_certificate_refuses_a_bad_rule():
    from solprop.modes import quadrature  # noqa: PLC0415 - private helper

    eta, weights = field_scale_rule(4)
    with pytest.raises(ModeError, match="invalid node or weight"):
        quadrature._certify(4, eta[::-1].copy(), weights, 7, 256)
    with pytest.raises(ModeError, match="misses moment 0"):
        quadrature._certify(4, eta, 1.01 * weights, 7, 256)


def test_a_radau_rule_that_loses_its_axis_node_is_refused(monkeypatch):
    from solprop.modes import quadrature  # noqa: PLC0415 - patched module

    field_scale_rule.cache_clear()
    monkeypatch.setattr(
        quadrature, "solve_banded", lambda *_args: np.zeros(_args[-1].shape)
    )
    try:
        with pytest.raises(ModeError, match="lost its axis node"):
            field_scale_rule(6, radau=True)
    finally:
        field_scale_rule.cache_clear()


def test_area_weights_need_a_whole_radial_only_rule():
    rule = quadrature_rule("polar", full=False, n_r=4, radial_rule="field_scale")
    with pytest.raises(ModeError, match="radial-only polar rule without panels"):
        dataclasses.replace(rule, full=True)
    with pytest.raises(ModeError, match="radial-only polar rule without panels"):
        dataclasses.replace(rule, breaks=(0.5,))

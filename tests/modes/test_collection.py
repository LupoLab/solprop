# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math
from typing import cast

import numpy as np
import pytest
from scipy.integrate import quad_vec

from solprop.exceptions import ModeError
from solprop.modes import ChannelLabel, CrossSection, MarcatiliMode, RectMode
from solprop.modes import collection as collection_module
from solprop.modes.collection import (
    ModeCollection,
    orthonormality,
    overlap,
    polarisation_pairs,
)
from solprop.modes.quadrature import mode_matrix, transverse_quadrature
from solprop.modes.slab import SlabMode
from solprop.modes.stepindex import StepIndexMode
from solprop.profiles import Linear

RADIUS = 125e-6
DISC: tuple[CrossSection, tuple[float, float], tuple[float, float]] = (
    "polar",
    (0.0, 0.0),
    (RADIUS, 2.0 * math.pi),
)


def he_modes(count=3):
    return [MarcatiliMode(RADIUS, "He", 1.0, m=m) for m in range(1, count + 1)]


class DummyMode:
    """The smallest thing that satisfies the Mode protocol, for the error paths."""

    z_constant = True
    scale_invariant = True
    azimuthal_order = 0
    channel_label = ChannelLabel("dummy", None, None, None, None, "dummy")

    def principal_component(self, z=0.0):
        return None

    def neff(self, omega, z=0.0):
        return np.ones_like(omega, dtype=complex)

    def neff_minus_one(self, omega, z=0.0):
        return np.zeros_like(omega, dtype=complex)

    def dneff_domega(self, omega, z=0.0):
        return np.zeros_like(omega, dtype=complex)

    def aeff(self, z=0.0):
        return 1.0

    def dimlimits(
        self, z=0.0
    ) -> tuple[CrossSection, tuple[float, float], tuple[float, float]]:
        return DISC

    def field(self, points, z=0.0):
        return np.ones((points.shape[0], 2))


# ---------------------------------------------------------------------- overlap


def test_a_mode_overlaps_itself_by_one():
    """Which is the statement that the package normalisation is what it claims."""
    for mode in he_modes(3):
        assert overlap(mode, mode) == pytest.approx(1.0, rel=1e-12)


def test_radially_distinct_modes_are_orthogonal():
    """The Bessel zeros make HE_1m an orthogonal family; dblquad's own tolerance is
    what sets the gate, not the modes."""
    modes = he_modes(4)
    for first in range(len(modes)):
        for second in range(first + 1, len(modes)):
            assert overlap(modes[first], modes[second]) == pytest.approx(0.0, abs=1e-9)


def test_a_high_order_mode_of_a_narrow_capillary_overlaps_converged():
    """The adaptive rule converges where the unnormalised integral is tiny.

    A 75 um capillary's HE1,24 has N = 2e-13 in SI units, below dblquad's default
    absolute tolerance: integrated unscaled, the rule stopped unconverged and
    silently, at 0.959 for the mode with itself and -1.1e-2 against HE1,23.
    """
    narrow = [MarcatiliMode(75e-6, "Ar", 3.0, m=m) for m in (23, 24)]
    assert overlap(narrow[1], narrow[1]) == pytest.approx(1.0, rel=1e-9)
    assert overlap(narrow[0], narrow[1]) == pytest.approx(0.0, abs=1e-9)


def test_the_two_polarisations_of_a_mode_are_orthogonal():
    mode = MarcatiliMode(RADIUS, "He", 1.0)
    partner = mode.orthogonal_polarisation
    assert partner is not None
    assert overlap(mode, partner) == pytest.approx(0.0, abs=1e-9)


def test_hybrid_and_transverse_families_are_orthogonal():
    hybrid = MarcatiliMode(RADIUS, "He", 1.0, n=2)
    transverse = MarcatiliMode(RADIUS, "He", 1.0, kind="TE", n=0)
    assert overlap(hybrid, transverse) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------- orthonormality


def test_orthonormality_is_the_identity_for_a_good_set():
    modes = he_modes(3)
    matrix = orthonormality(modes)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, np.eye(3), rtol=0.0, atol=1e-9)


def test_orthonormality_is_symmetric():
    modes = [
        MarcatiliMode(RADIUS, "He", 1.0),
        MarcatiliMode(RADIUS, "He", 1.0, n=2),
        MarcatiliMode(RADIUS, "He", 1.0, kind="TM", n=0),
    ]
    matrix = orthonormality(modes)
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=0.0)


def pairwise(modes):
    """Return the overlap matrix pair by pair, the adaptive reference."""
    return np.array([[overlap(first, second) for second in modes] for first in modes])


def test_the_matrix_is_the_pairwise_overlaps_on_a_polar_set():
    """HE, EH-like hybrid, TE and TM modes and a polarisation partner: the one-pass
    matrix against dblquad pair by pair, to dblquad's accuracy on these products."""
    modes = [
        *he_modes(2),
        MarcatiliMode(RADIUS, "He", 1.0, n=2),
        MarcatiliMode(RADIUS, "He", 1.0, kind="TE", n=0),
        MarcatiliMode(RADIUS, "He", 1.0, kind="TM", n=0),
    ]
    partner = modes[0].orthogonal_polarisation
    assert partner is not None
    modes.append(partner)
    np.testing.assert_allclose(
        orthonormality(modes), pairwise(modes), rtol=0.0, atol=1e-9
    )


def test_the_matrix_is_the_pairwise_overlaps_on_cartesian_and_slab_sets():
    """A rectangular guide's two polarisations of four orders, and four slab orders."""
    rectangular = [
        RectMode(50e-6, 12.5e-6, n=n, m=m, polarisation=polarisation)
        for n, m in ((1, 1), (1, 2), (2, 1), (2, 2))
        for polarisation in ("x", "y")
    ]
    slab = [SlabMode(63.5e-6, order=order) for order in range(1, 5)]
    for modes in (rectangular, slab):
        np.testing.assert_allclose(
            orthonormality(modes), pairwise(modes), rtol=0.0, atol=1e-9
        )


@pytest.mark.medium
def test_many_high_order_capillary_modes_are_orthonormal():
    """24 HE1m in two polarisations of a 75 um capillary, in seconds.

    Pair by pair this took minutes, and before the normalisation went inside the
    integrand it refused the set: its HE1,24 integral, 2e-13 in SI units, was below
    dblquad's absolute tolerance. 1e-12 is the matrix's own tolerance.
    """
    modes = ModeCollection.with_polarisations(
        [MarcatiliMode(75e-6, "Ar", 3.0, m=m) for m in range(1, 25)]
    )
    np.testing.assert_allclose(modes.orthonormality(), np.eye(48), rtol=0.0, atol=1e-12)


def test_a_high_azimuthal_order_does_not_alias_into_a_false_overlap():
    """HE with n = 64 against TM01: their product carries cos(64 theta), which the
    16- and 32-node offset rules both read as +1 at every node. Started there, the two
    rules agreed on an overlap of 0.16; started above the products' harmonics (128
    nodes, then 256 to show convergence), the pair is orthogonal."""
    modes = [
        MarcatiliMode(RADIUS, "He", 1.0, n=64),
        MarcatiliMode(RADIUS, "He", 1.0, kind="TM", n=0),
    ]
    assert collection_module._first_nodes(modes) == 128
    np.testing.assert_allclose(orthonormality(modes), np.eye(2), rtol=0.0, atol=1e-9)


def test_a_first_rule_at_the_cap_is_still_refined(monkeypatch):
    """A polar set whose first rule reaches the cap still gets two doublings: with the
    cap at 128, the n = 64 / TM01 pair starts there and converges on 256 nodes, where
    a single rule had nothing to compare against and was refused."""
    monkeypatch.setattr(collection_module, "_MAX_NODES", 128)
    modes = [
        MarcatiliMode(RADIUS, "He", 1.0, n=64),
        MarcatiliMode(RADIUS, "He", 1.0, kind="TM", n=0),
    ]
    np.testing.assert_allclose(orthonormality(modes), np.eye(2), rtol=0.0, atol=1e-9)


def test_orthonormality_refuses_a_rule_that_does_not_converge(monkeypatch):
    """A rule whose answer keeps moving on doubling is refused at the cap."""
    rule = collection_module._second_rule

    def drifting(kind, limits, nodes):
        points, weights = rule(kind, limits, nodes)
        return points, weights * (1.0 + 1.0 / nodes)

    monkeypatch.setattr(collection_module, "_second_rule", drifting)
    with pytest.raises(ModeError, match="did not converge"):
        orthonormality(he_modes(2))


def test_orthonormality_refuses_an_unconverged_first_coordinate(monkeypatch):
    """quad_vec stops short silently; the matrix it returns is refused, not used."""

    def starved(*args, **kwargs):
        return quad_vec(*args, **kwargs, limit=1)

    monkeypatch.setattr(collection_module, "quad_vec", starved)
    with pytest.raises(ModeError, match="first coordinate"):
        orthonormality(he_modes(2))


def test_the_fixed_rule_reproduces_the_adaptive_overlaps():
    """The two ways of asking the same question, which is the point of having both.

    The adaptive integral is the reference the fixed rule is judged against; if the
    quadrature tests used the rule for both sides they would be tautologies. They
    agree to 1e-9, which is dblquad's own tolerance on an oscillatory Bessel product,
    not the rule's -- the rule is exact here.
    """
    modes = [
        MarcatiliMode(RADIUS, "He", 1.0, m=1),
        MarcatiliMode(RADIUS, "He", 1.0, m=2),
        MarcatiliMode(RADIUS, "He", 1.0, n=2),
    ]
    rule = transverse_quadrature(modes, n_r=32, n_theta=16)
    matrices = mode_matrix(modes, rule)
    # The projection carries the weights and the basis normalisation, so the discrete
    # inner product is the normalised overlap itself: the identity on this set.
    discrete = matrices.projection @ matrices.synthesis
    np.testing.assert_allclose(discrete, orthonormality(modes), rtol=0.0, atol=1e-9)


# ------------------------------------------------------------ polarisation pairs


def test_each_hybrid_mode_gains_a_partner_in_a_fixed_order():
    modes = he_modes(2)
    pairs = polarisation_pairs(modes)
    assert len(pairs) == 4
    assert pairs[0] is modes[0]
    assert pairs[2] is modes[1]
    partner = pairs[1]
    assert isinstance(partner, MarcatiliMode)
    assert partner.phi == pytest.approx(math.pi / 2, rel=1e-15)
    assert partner.m == modes[0].m


def test_a_transverse_mode_is_carried_once():
    """TE_0m and TM_0m are different modes, not two polarisation states of one."""
    modes = [
        MarcatiliMode(RADIUS, "He", 1.0, kind="TE", n=0),
        MarcatiliMode(RADIUS, "He", 1.0, kind="TM", n=0),
    ]
    assert polarisation_pairs(modes) == tuple(modes)


def test_the_partner_of_a_higher_order_hybrid_turns_by_a_quarter_of_its_period():
    """The field's azimuthal phase is n (theta + phi), so a quarter turn is pi/(2n)."""
    mode = MarcatiliMode(RADIUS, "He", 1.0, n=3)
    partner = mode.orthogonal_polarisation
    assert partner is not None
    assert partner.phi == pytest.approx(math.pi / 6, rel=1e-15)
    assert overlap(mode, partner) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------- the collection


def test_a_collection_is_an_ordered_sequence():
    modes = he_modes(3)
    collection = ModeCollection(modes)
    assert len(collection) == 3
    assert collection[1] is modes[1]
    assert list(collection) == modes


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: overlap(
                MarcatiliMode(RADIUS, "He", 1.0), MarcatiliMode(2 * RADIUS, "He", 1.0)
            ),
            "share a cross-section",
        ),
        (lambda: overlap(DummyMode(), DummyMode()), "provides no normalisation"),
        (
            lambda: orthonormality(
                [MarcatiliMode(RADIUS, "He", 1.0), MarcatiliMode(2 * RADIUS, "He", 1.0)]
            ),
            "share a cross-section",
        ),
        (lambda: orthonormality([]), "at least one mode"),
        (lambda: polarisation_pairs([DummyMode()]), "orthogonal polarisation"),
        (lambda: ModeCollection(()), "at least one mode"),
        # Two *different* channels, so that it is the cross-section that is refused
        # and not the repeated channel a collection also refuses.
        (
            lambda: ModeCollection(
                [
                    MarcatiliMode(RADIUS, "He", 1.0),
                    MarcatiliMode(2 * RADIUS, "He", 1.0, m=2),
                ]
            ),
            "share a cross-section",
        ),
        (lambda: ModeCollection.he1m(RADIUS, "He", 0.8, count=0), "at least one mode"),
    ],
)
def test_a_set_that_is_not_one_guide_is_refused(call, message):
    with pytest.raises(ModeError, match=message):
        call()


def test_a_collection_reports_the_shared_cross_section():
    collection = ModeCollection(he_modes(2))
    assert collection.dimlimits() == (
        "polar",
        (0.0, 0.0),
        (RADIUS, 2.0 * math.pi),
    )


def test_a_collection_catches_a_disagreement_that_only_appears_along_z():
    """One mode tapered and another not is valid at z = 0 and wrong everywhere else."""
    collection = ModeCollection(
        [
            MarcatiliMode(Linear(RADIUS, 75e-6, 1.0), "He", 1.0),
            MarcatiliMode(RADIUS, "He", 1.0, m=2),
        ]
    )
    assert collection.dimlimits(0.0)[2][0] == pytest.approx(RADIUS)
    with pytest.raises(ModeError, match="share a cross-section at z = 1 m"):
        collection.dimlimits(1.0)


@pytest.mark.parametrize(
    ("modes", "order", "full"),
    [
        ([("HE", 1, 1)], 0, False),
        ([("HE", 1, 1), ("HE", 1, 3)], 0, False),
        ([("HE", 1, 1), ("HE", 2, 1)], 1, True),
        ([("HE", 1, 1), ("TE", 0, 1)], 1, True),
        ([("HE", 4, 1)], 3, True),
    ],
)
def test_a_collection_reports_what_the_rule_needs(modes, order, full):
    collection = ModeCollection(
        [MarcatiliMode(RADIUS, "He", 1.0, kind=kind, n=n, m=m) for kind, n, m in modes]
    )
    assert collection.azimuthal_order == order
    assert collection.needs_full_rule is full


def test_a_collection_reports_the_traits_of_its_members():
    fixed = ModeCollection(he_modes(2))
    assert fixed.z_constant
    assert fixed.scale_invariant
    tapered = ModeCollection(
        [
            MarcatiliMode(Linear(RADIUS, 75e-6, 1.0), "He", 1.0),
            MarcatiliMode(Linear(RADIUS, 75e-6, 1.0), "He", 1.0, m=2),
        ]
    )
    assert not tapered.z_constant
    assert tapered.scale_invariant


def test_a_collection_can_be_built_with_both_polarisations():
    collection = ModeCollection.with_polarisations(he_modes(2))
    assert len(collection) == 4
    members = [mode for mode in collection if isinstance(mode, MarcatiliMode)]
    assert len(members) == 4
    assert [mode.m for mode in members] == [1, 1, 2, 2]
    assert [round(mode.phi, 6) for mode in members] == [0.0, 1.570796, 0.0, 1.570796]


def test_a_collection_measures_its_own_orthonormality():
    collection = ModeCollection(he_modes(3))
    np.testing.assert_allclose(collection.orthonormality(), np.eye(3), atol=1e-9)


@pytest.mark.medium
def test_a_vector_collection_is_orthonormal():
    """Both polarisations of two radial orders: four modes that must not overlap."""
    collection = ModeCollection.with_polarisations(he_modes(2))
    np.testing.assert_allclose(collection.orthonormality(), np.eye(4), atol=1e-9)


def test_a_collection_feeds_the_quadrature_and_the_matrices_directly():
    """It is a Sequence[Mode], so nothing has to unwrap it."""
    collection = ModeCollection(he_modes(3))
    rule = transverse_quadrature(collection, n_r=24)
    assert not rule.full
    matrices = mode_matrix(collection, rule, components=("x",))
    assert matrices.synthesis.shape == (24, 3)


def he1m_modes(*arguments, **keywords) -> tuple[MarcatiliMode, ...]:
    """Return an he1m set as the capillary modes it is built from."""
    return cast(
        "tuple[MarcatiliMode, ...]",
        tuple(ModeCollection.he1m(*arguments, **keywords).modes),
    )


def test_he1m_builds_the_first_count_hybrid_modes():
    modes = he1m_modes(RADIUS, "He", 0.8, count=4)
    assert len(modes) == 4
    assert [mode.m for mode in modes] == [1, 2, 3, 4]
    assert {(mode.kind, mode.n) for mode in modes} == {("HE", 1)}


def test_he1m_needs_no_azimuthal_rule():
    # |J_0(u r/a)| does not depend on theta, so the set integrates radially.
    modes = ModeCollection.he1m(RADIUS, "He", 0.8, count=3)
    assert modes.azimuthal_order == 0
    assert not modes.needs_full_rule


def test_he1m_is_orthonormal():
    # The adaptive overlaps, not the fixed rule the run uses: a set checked on its own
    # quadrature would be orthonormal by construction.
    got = ModeCollection.he1m(RADIUS, "He", 0.8, count=3).orthonormality()
    # dblquad's own accuracy on the Bessel products, measured at 4e-11 here.
    np.testing.assert_allclose(got, np.eye(3), atol=1e-9)


def test_he1m_carries_the_mode_keywords_to_every_mode():
    modes = he1m_modes(
        RADIUS, "He", 0.8, count=2, model="reduced", loss_scale=0.0, cladding=1.45
    )
    assert [mode.model for mode in modes] == ["reduced", "reduced"]
    assert [mode.loss_scale for mode in modes] == [0.0, 0.0]
    assert [mode.cladding_index for mode in modes] == [1.45, 1.45]


def test_he1m_matches_the_modes_built_by_hand():
    by_hand = [MarcatiliMode(RADIUS, "He", 0.8, m=m, loss_scale=0.0) for m in (1, 2)]
    got = he1m_modes(RADIUS, "He", 0.8, count=2, loss_scale=0.0)
    omega = np.array([2.356e15])
    for one, other in zip(got, by_hand, strict=True):
        np.testing.assert_array_equal(one.neff(omega), other.neff(omega))


# ----------------------------------------------------- step-index channels of one guide


def stepindex_pair(numerical_aperture=False):
    """HE11 and HE12 of one fibre: each bounds the cladding at its own outer radius."""
    if numerical_aperture:
        return [
            StepIndexMode.from_numerical_aperture(
                10e-6, 0.12, m=m, profile_wavelength=1e-6
            )
            for m in (1, 2)
        ]
    return [
        StepIndexMode(10e-6, core=1.46, cladding=1.44, m=m, profile_wavelength=1e-6)
        for m in (1, 2)
    ]


@pytest.mark.parametrize("numerical_aperture", [False, True])
def test_step_index_channels_of_one_guide_form_a_collection(numerical_aperture):
    modes = stepindex_pair(numerical_aperture)
    assert modes[0].dimlimits()[2][0] != modes[1].dimlimits()[2][0]
    assert modes[0].same_guide(modes[1])
    collection = ModeCollection(modes)
    # The widest disc, over which both channels' fields are exact.
    widest = max(mode.dimlimits()[2][0] for mode in modes)
    assert collection.dimlimits()[2][0] == widest


def test_step_index_channels_of_one_guide_have_overlaps_on_the_widest_disc():
    modes = stepindex_pair()
    got = ModeCollection(modes).orthonormality()
    # Each channel normalised to itself, to the integration tolerance (1e-12); the
    # off-diagonal is the vector modes' small non-orthogonality (step-index guide).
    np.testing.assert_allclose(np.diag(got), 1.0, rtol=0.0, atol=1e-10)
    assert got[0, 1] == got[1, 0]
    assert 0.0 < abs(got[0, 1]) < 1e-2
    assert overlap(modes[0], modes[1]) == pytest.approx(got[0, 1], abs=1e-12)
    rule = transverse_quadrature(modes, n_r=48)
    assert mode_matrix(modes, rule, components=("x",)).synthesis.shape[1] == 2


def test_step_index_channels_of_different_guides_are_refused():
    first = StepIndexMode(10e-6, core=1.46, cladding=1.44, profile_wavelength=1e-6)
    other = StepIndexMode(10e-6, core=1.46, cladding=1.43, m=2, profile_wavelength=1e-6)
    assert not first.same_guide(other)
    with pytest.raises(ModeError, match="share a cross-section"):
        ModeCollection([first, other])

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math
from typing import cast

import numpy as np
import pytest

from solprop.exceptions import ModeError
from solprop.modes import ChannelLabel, CrossSection, MarcatiliMode
from solprop.modes.collection import (
    ModeCollection,
    orthonormality,
    overlap,
    polarisation_pairs,
)
from solprop.modes.quadrature import mode_matrix, transverse_quadrature
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


def test_the_two_polarisations_of_a_mode_are_orthogonal():
    mode = MarcatiliMode(RADIUS, "He", 1.0)
    partner = mode.orthogonal_polarisation
    assert partner is not None
    assert overlap(mode, partner) == pytest.approx(0.0, abs=1e-9)


def test_hybrid_and_transverse_families_are_orthogonal():
    hybrid = MarcatiliMode(RADIUS, "He", 1.0, n=2)
    transverse = MarcatiliMode(RADIUS, "He", 1.0, kind="TE", n=0)
    assert overlap(hybrid, transverse) == pytest.approx(0.0, abs=1e-9)


def test_an_overlap_needs_one_cross_section():
    with pytest.raises(ModeError, match="needs one cross-section"):
        overlap(MarcatiliMode(RADIUS, "He", 1.0), MarcatiliMode(2 * RADIUS, "He", 1.0))


def test_an_overlap_needs_a_closed_form_normalisation():
    with pytest.raises(ModeError, match="provides no normalisation"):
        overlap(DummyMode(), DummyMode())


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


def test_orthonormality_needs_a_mode():
    with pytest.raises(ModeError, match="at least one mode"):
        orthonormality([])


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


def test_a_mode_that_cannot_say_is_refused():
    with pytest.raises(ModeError, match="orthogonal polarisation"):
        polarisation_pairs([DummyMode()])


# ------------------------------------------------------------- the collection


def test_a_collection_is_an_ordered_sequence():
    modes = he_modes(3)
    collection = ModeCollection(modes)
    assert len(collection) == 3
    assert collection[1] is modes[1]
    assert list(collection) == modes


def test_a_collection_needs_a_mode():
    with pytest.raises(ModeError, match="at least one mode"):
        ModeCollection(())


def test_a_collection_refuses_mismatched_cross_sections():
    # Two *different* channels, so that it is the cross-section that is refused
    # and not the repeated channel a collection also refuses.
    with pytest.raises(ModeError, match="share a cross-section"):
        ModeCollection(
            [
                MarcatiliMode(RADIUS, "He", 1.0),
                MarcatiliMode(2 * RADIUS, "He", 1.0, m=2),
            ]
        )


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
    with pytest.raises(ModeError, match="disagree about the cross-section"):
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
    modes = he1m_modes(RADIUS, "He", 0.8, 4)
    assert len(modes) == 4
    assert [mode.m for mode in modes] == [1, 2, 3, 4]
    assert {(mode.kind, mode.n) for mode in modes} == {("HE", 1)}


def test_he1m_needs_no_azimuthal_rule():
    # |J_0(u r/a)| does not depend on theta, so the set integrates radially.
    modes = ModeCollection.he1m(RADIUS, "He", 0.8, 3)
    assert modes.azimuthal_order == 0
    assert not modes.needs_full_rule


def test_he1m_is_orthonormal():
    # The adaptive overlaps, not the fixed rule the run uses: a set checked on its own
    # quadrature would be orthonormal by construction.
    got = ModeCollection.he1m(RADIUS, "He", 0.8, 3).orthonormality()
    # dblquad's own accuracy on the Bessel products, measured at 4e-11 here.
    np.testing.assert_allclose(got, np.eye(3), atol=1e-9)


def test_he1m_carries_the_mode_keywords_to_every_mode():
    modes = he1m_modes(RADIUS, "He", 0.8, 2, model="reduced", loss=False, cladding=1.45)
    assert [mode.model for mode in modes] == ["reduced", "reduced"]
    assert [mode.loss for mode in modes] == [False, False]
    assert [mode.cladding_index for mode in modes] == [1.45, 1.45]


def test_he1m_matches_the_modes_built_by_hand():
    by_hand = [MarcatiliMode(RADIUS, "He", 0.8, m=m, loss=False) for m in (1, 2)]
    got = he1m_modes(RADIUS, "He", 0.8, 2, loss=False)
    omega = np.array([2.356e15])
    for one, other in zip(got, by_hand, strict=True):
        np.testing.assert_array_equal(one.neff(omega), other.neff(omega))


def test_he1m_refuses_an_empty_set():
    with pytest.raises(ModeError, match="at least one mode"):
        ModeCollection.he1m(RADIUS, "He", 0.8, 0)

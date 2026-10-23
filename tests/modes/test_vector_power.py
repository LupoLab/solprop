# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check analytic production fields against independent longitudinal matching."""

import importlib
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modal_oracles import register_modal_oracles
from modal_provider_cases import OMEGA, make_provider
from solprop.exceptions import ModeError
from solprop.modes import MarcatiliMode, ModeCollection
from solprop.modes.vector import ModalNodeRule, flux_product


@pytest.mark.parametrize(
    "family,n", [("HE", 1), ("EH", 1), ("TE", 0), ("TM", 0), ("HE", 2)]
)
@pytest.mark.parametrize("parity", ["even", "odd"])
def test_stepindex_maxwell_boundary_residual(family, n, parity):
    # Predeclared engineering bound, independently of observed defects.
    bound = 1e-9
    provider = make_provider(descriptors=((family, n, 1, parity),))
    mode = provider.modes[0]
    register_modal_oracles()
    reference = importlib.import_module("modal_design.vector_modes")
    longitudinal = reference.solve_boundary(mode)
    rule = provider.node_rule(OMEGA)
    sample = provider.solve(OMEGA, 0.0, nodes=rule)
    public_e, public_h = mode.electromagnetic_field(rule.xy, OMEGA)
    # The public E/H API uses the same full-domain norm as the provider.
    assert_allclose(public_e, sample.electric[:, :, 0], rtol=0, atol=0)
    assert_allclose(public_h, sample.magnetic[:, :, 0], rtol=0, atol=0)
    r, theta = (
        np.hypot(rule.xy[:, 0], rule.xy[:, 1]),
        np.arctan2(rule.xy[:, 1], rule.xy[:, 0]),
    )
    # The oracle uses optics phasors. This is a reference comparison, not a new
    # production spectrum-conversion boundary.
    e, h = reference.sampled(longitudinal, r, theta, parity)
    power = reference.power(e, h, rule.weights).real
    e, h = e.conj() / np.sqrt(power), h.conj() / np.sqrt(power)
    overlap = np.vdot(e, sample.electric[:, :, 0])
    phase = overlap / abs(overlap)
    assert (
        np.linalg.norm(sample.electric[:, :, 0] - phase * e) / np.linalg.norm(e) < bound
    )
    assert (
        np.linalg.norm(sample.magnetic[:, :, 0] - phase * h) / np.linalg.norm(h) < bound
    )
    assert sample.boundary_residual[0] < bound
    angles = np.arange(25) * 2 * np.pi / 25
    radial = np.column_stack((np.cos(angles), np.sin(angles)))
    tangent = np.column_stack((-np.sin(angles), np.cos(angles)))
    inside, hin = mode.electromagnetic_field(radial * 1.5e-6 * (1 - 1e-12), OMEGA)
    outside, hout = mode.electromagnetic_field(radial * 1.5e-6 * (1 + 1e-12), OMEGA)
    e_scale, h_scale = np.max(abs(inside)), np.max(abs(hin))
    assert (
        np.max(abs(np.sum((1.45**2 * inside[:, :2] - outside[:, :2]) * radial, axis=1)))
        < bound * e_scale
    )
    assert (
        np.max(abs(np.sum((hin[:, :2] - hout[:, :2]) * radial, axis=1)))
        < bound * h_scale
    )
    assert (
        np.max(abs(np.sum((inside[:, :2] - outside[:, :2]) * tangent, axis=1)))
        < bound * e_scale
    )
    assert np.max(abs(hin - hout)) < bound * h_scale
    axis_e, axis_h = mode.electromagnetic_field(np.zeros((1, 2)), OMEGA)
    assert np.all(np.isfinite(axis_e)) and np.all(np.isfinite(axis_h))


def test_power_orthogonality():
    bound = 1e-9
    descriptors = (
        ("HE", 1, 1, "even"),
        ("HE", 1, 1, "odd"),
        ("HE", 1, 2, "even"),
        ("EH", 1, 1, "even"),
        ("TE", 0, 1, "even"),
        ("TM", 0, 1, "even"),
    )
    provider = make_provider(descriptors=descriptors)
    rule = provider.node_rule(OMEGA)
    sample = provider.solve(OMEGA, 0.0, nodes=rule)
    assert_allclose(sample.flux_metric, np.eye(len(descriptors)), rtol=0, atol=bound)
    # Unit full-domain power is independent of the caller's deliberately truncated rule.
    mask = rule.region_index == 0
    short = ModalNodeRule(
        rule.xy[mask], rule.weights[mask], rule.region_index[mask], {}
    )
    truncated = provider.solve(OMEGA, 0.0, nodes=short)
    assert np.all(np.real(np.diag(truncated.flux_metric)) < 1)


def test_channel_labels_and_partner_validation():
    provider = make_provider()
    mode = provider.modes[0]
    # The label is each mode's own, not an isinstance ladder's guess from its class.
    capillary = ModeCollection([MarcatiliMode(125e-6, "He", 1.0)]).channel_labels[0]
    assert (capillary.family, capillary.radial_order) == ("HE", 1)
    assert dict(capillary.as_mapping())["display_label"] == "HE1,1:even"
    collection = ModeCollection.with_polarisations([mode])
    assert [label.parity for label in collection.channel_labels] == ["even", "odd"]
    assert (
        collection.channel_labels[0].display_label
        != collection.channel_labels[1].display_label
    )
    # A rotation that is neither the mode nor its quarter-turn partner is a gauge,
    # not a parity, and says so instead of claiming to be the even one.
    # The even member is x-polarised in both families, and the definition is
    # physical: the field on axis. Step-index "even" HE11 builds itself from
    # cos(theta) on the radial component and a capillary "even" HE11 is the
    # unrotated mode (ADR-031, docs/dev/decisions.md), and both are x-polarised there.
    on_axis = np.array([[0.0, 0.0], [1e-9, 0.3], [1e-9, 1.1]])
    unrotated = MarcatiliMode(125e-6, "He", 1.0)
    partner = unrotated.orthogonal_polarisation
    assert partner is not None
    assert partner.channel_label.parity == "odd"
    exact = make_provider(descriptors=(("HE", 1, 1, "even"),)).modes[0]
    for even in (exact, unrotated):
        assert even.channel_label.parity == "even"
        field = even.field(on_axis)
        assert np.all(abs(field[:, 0]) > 0)
        # Dominant is not enough: the orthogonal component of an even HE11 is zero
        # on the axis and 1e-8 of the principal one a nanometre off it.
        assert np.all(abs(field[:, 1]) < 1e-7 * abs(field[:, 0]))
    rotated = MarcatiliMode(125e-6, "He", 1.0).rotated(0.3)
    assert rotated.channel_label.parity is None
    assert rotated.channel_label.display_label == "HE1,1:phi=0.3"
    assert MarcatiliMode(125e-6, kind="TE", n=0).channel_label.parity is None
    with pytest.raises(ModeError, match="duplicate physical channel"):
        ModeCollection([mode, mode])
    te = replace(mode, kind="TE", n=0)
    with pytest.raises(ModeError, match="no parity partner"):
        ModeCollection([te, replace(te, parity="odd")])


@pytest.mark.parametrize("reverse", [False, True])
def test_collection_covers_every_stepindex_cladding_tail(reverse):
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("TE", 0, 1, "even")))
    modes = tuple(reversed(provider.modes)) if reverse else provider.modes
    limits = [mode.dimlimits() for mode in modes]
    assert limits[0][2][0] != limits[1][2][0]
    # Selecting a second physical channel must not truncate either tail; the
    # integration domain is independent of channel order.
    assert ModeCollection(modes).dimlimits() == (
        limits[0][0],
        limits[0][1],
        (max(item[2][0] for item in limits), limits[0][2][1]),
    )


def test_full_fields_refuse_unqualified_loss_and_invalid_points():
    mode = make_provider().modes[0]
    with pytest.raises(ModeError, match="leaky"):
        replace(mode, loss=1.0).electromagnetic_field(np.zeros((1, 2)), OMEGA)
    with pytest.raises(ModeError, match="finite Cartesian"):
        mode.electromagnetic_field(np.full((1, 2), np.nan), OMEGA)
    with pytest.raises(ModeError, match="omega"):
        mode.electromagnetic_field(np.zeros((1, 2)), 0.0)


def test_flux_product_coordinates():
    e = np.array([[[1, 1j], [0, 0], [0, 0]]], complex)
    h = np.array([[[0, 0], [1, 1j], [0, 0]]], complex)
    result = flux_product(e, h, np.ones(1))
    # Two cross products and their Hermitian average: a 16-epsilon bound.
    assert_allclose(
        result, np.array([[1, 1j], [-1j, 1]]) / 2, rtol=0, atol=16 * np.finfo(float).eps
    )

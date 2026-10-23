# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise tracked physical coordinates and held-out dense interpolation."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import CubicSpline

from bounds import EPS, assert_within
from modal_provider_cases import OMEGA, make_provider
from solprop.constants import C
from solprop.exceptions import ModeError
from solprop.grid import EnvelopeGrid
from solprop.linear import Frame, dispersion_operator
from solprop.modes import MarcatiliMode, StepIndexMode
from solprop.modes.base import alpha, beta1, beta2
from solprop.modes.graded_index import GradedIndexProvider, PowerLawPermittivity
from solprop.modes.guide import (
    BASIS_UNITARY_ATOL,
    ChannelBasis,
    ModalGuide,
    ProviderChannelMode,
    StepIndexProvider,
    circular_pair,
    provider_channels,
    radial_node_rule,
)
from solprop.modes.material import IndexMaterial, MaterialRegion
from solprop.modes.protocol import ChannelLabel
from solprop.modes.tabulated import _SampledIndexTable, sample_index_table
from solprop.modes.vector import ModalSample
from solprop.profiles import Linear
from solprop.translate.modal_tables import (
    DEFAULT_GRAM_ATOL,
    MAX_GENERATOR_COUPLING,
    MIN_OVERLAP_DOMINANCE,
    DenseModalTable,
    _channel_index_table,
    _check_transformed_passivity,
    _overlap,
    align_modal_sample,
    apply_channel_basis,
    channel_area,
    composed_gauge,
    sample_modal_table,
    sample_provider_table,
    solved_in_table_basis,
    validate_modal_table,
)

GRIN_OMEGA = 2 * np.pi * C / 1.03e-6
"""The graded-index reference frequency, beside the strand's OMEGA."""


class RotatingProvider:
    """Rotate a provider's declared degenerate pair smoothly with frequency.

    The physical guide is unchanged; only the gauge the provider hands back
    rotates, as a real solver's arbitrary eigenbasis does inside a degenerate
    subspace. The rotation is smooth in omega so that a tracked table can
    interpolate, and its starting angle is seeded so that nothing depends on the
    first knot happening to be the identity.
    """

    def __init__(self, base, reference, seed, rate=35.0):
        self._base, self._reference, self._rate = base, reference, rate
        self._offset = float(np.random.default_rng(seed).uniform(0, 2 * np.pi))

    # Declared, not delegated through __getattr__: since Python 3.12 an isinstance
    # check against a runtime-checkable protocol uses inspect.getattr_static, which
    # does not call __getattr__, so a dynamic wrapper is not a ModalProvider.
    @property
    def regions(self):
        return self._base.regions

    @property
    def z_breakpoints(self):
        return self._base.z_breakpoints

    @property
    def z_domain(self):
        return self._base.z_domain

    @property
    def omega_domain(self):
        return self._base.omega_domain

    @property
    def identity(self):
        return self._base.identity

    def _rotation(self, omega):
        angle = self._offset + self._rate * (omega / self._reference - 1)
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)

    def node_rule(self, omega, z=0.0, *, n_r=None, n_theta=None):
        return self._base.node_rule(omega, z, n_r=n_r, n_theta=n_theta)

    def solve(self, omega, z, *, nodes):
        sample = self._base.solve(omega, z, nodes=nodes)
        rotation = self._rotation(omega)
        return replace(
            sample,
            electric=sample.electric @ rotation,
            magnetic=sample.magnetic @ rotation,
            generator=rotation.conj().T @ sample.generator @ rotation,
            flux_metric=rotation.conj().T @ sample.flux_metric @ rotation,
            backward_electric=sample.backward_electric @ rotation.conj(),
            backward_magnetic=sample.backward_magnetic @ rotation.conj(),
            reciprocal_gram=rotation.conj().T @ sample.reciprocal_gram @ rotation,
        )


def tapered_provider(*, n_r):
    """Return a provider on a small physical taper and a common node rule."""
    original = make_provider(radius=250e-9)
    # A small physical taper keeps the common quadrature ownership fixed. Tables
    # spanning an interface crossing must be split or use a different common rule.
    path = Linear(250e-9, 250.0025e-9, 1.0)
    provider = StepIndexProvider(
        (replace(original.modes[0], radius=path),),
        (
            MaterialRegion("core", 0.0, path, original.regions[0].material),
            MaterialRegion("cladding", path, None, original.regions[1].material),
        ),
    )
    return provider, provider.node_rule(OMEGA, n_r=n_r)


def test_phase_and_degenerate_tracking():
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = provider.node_rule(OMEGA, n_r=32)
    sample = provider.solve(OMEGA, 0.0, nodes=nodes)
    assert sample.backward_electric is not None
    assert sample.backward_magnetic is not None
    assert sample.reciprocal_gram is not None
    rng = np.random.default_rng(4103)
    for _ in range(4):
        rotation = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[
            0
        ]
        candidate = replace(
            sample,
            electric=sample.electric @ rotation,
            magnetic=sample.magnetic @ rotation,
            generator=rotation.conj().T @ sample.generator @ rotation,
            flux_metric=rotation.conj().T @ sample.flux_metric @ rotation,
            backward_electric=sample.backward_electric @ rotation.conj(),
            backward_magnetic=sample.backward_magnetic @ rotation.conj(),
            reciprocal_gram=rotation.conj().T @ sample.reciprocal_gram @ rotation,
        )
        aligned = align_modal_sample(
            sample, candidate, nodes, degenerate_blocks=((0, 1),)
        )
        # Four physical overlap contractions and a two-dimensional SVD.
        bound = 128 * np.finfo(float).eps * len(nodes.xy)
        assert_allclose(
            aligned.electric,
            sample.electric,
            rtol=bound,
            atol=bound * np.max(abs(sample.electric)),
        )
        assert_allclose(
            aligned.generator,
            sample.generator,
            rtol=bound,
            atol=bound * np.max(abs(sample.generator)),
        )
        with pytest.raises(ModeError, match="disjoint"):
            align_modal_sample(
                sample, candidate, nodes, degenerate_blocks=((0, 1), (0, 1))
            )


def test_off_grid_profiles_and_dispersion():
    provider = make_provider(radius=250e-9)
    nodes = provider.node_rule(OMEGA, n_r=48)
    frequencies = np.linspace(0.98 * OMEGA, 1.02 * OMEGA, 9)
    table = sample_modal_table(provider, frequencies, np.array([0.0]), nodes)
    points = np.column_stack(((frequencies[:-1] + frequencies[1:]) / 2, np.zeros(8)))
    # Fixed engineering budgets: 1e-6 profile norm, 0.1 rad/m beta, 1e-6 on the
    # interpolated basis flux defect and 1e-6 relative beta1. Cubic interpolation
    # is tested away from a cutoff. Measured at these eight midpoints: 4.13e-9,
    # 1.28e-3 and 2.80e-9, so every budget keeps at least two orders of margin and
    # the flux column stays well above the rule's own Gram error at n_r=48.
    errors = validate_modal_table(
        table, provider, points, profile_rtol=1e-6, generator_atol=0.1, flux_atol=1e-6
    )
    assert errors.shape == (8, 3)
    mode = provider.modes[0]
    midpoint = table.evaluate(float(points[0, 0]), 0.0)
    assert midpoint.reciprocal_gram is not None
    assert_allclose(
        midpoint.reciprocal_gram,
        -4 * midpoint.flux_metric,
        rtol=64 * np.finfo(float).eps * len(nodes.xy),
        atol=0,
    )
    for frequency in points[:, 0]:
        exact = (
            mode.neff(np.array([frequency]))
            + frequency * mode.dneff_domega(np.array([frequency]))
        ) / C
        assert_allclose(
            table.generator_derivative(float(frequency), 0.0),
            -1j * exact.reshape(1, 1),
            rtol=1e-6,
            atol=0,
        )
    with pytest.raises(ModeError, match="extrapolate"):
        table.evaluate(1.03 * OMEGA, 0.0)
    with pytest.raises(ModeError, match="budget"):
        validate_modal_table(
            table,
            provider,
            points,
            profile_rtol=1e-15,
            generator_atol=1e-15,
            flux_atol=1e-15,
        )


def test_index_sampler_keeps_analytic_knot_derivatives():
    mode = make_provider().modes[0]
    frequencies = np.linspace(0.95 * OMEGA, 1.05 * OMEGA, 9)
    table = sample_index_table(mode, frequencies[::-1])
    # Hermite knots share the source values/slopes, up to one refractivity sum.
    assert_allclose(
        table.neff(frequencies),
        mode.neff(frequencies),
        rtol=8 * np.finfo(float).eps,
        atol=0,
    )
    # This mode's dneff/domega is about 1.4e-17 s/rad across the band, so 64 eps
    # of the largest knot value is 1.9e-31; the atol is five times that and is
    # there only so the comparison does not turn on a knot being exactly nonzero.
    # Measured, the knots reproduce the source exactly (0 absolute, 1.1e-16
    # relative), because the Hermite interpolant is built from these very slopes.
    assert_allclose(
        table.dneff_domega(frequencies),
        mode.dneff_domega(frequencies),
        rtol=64 * np.finfo(float).eps,
        atol=1e-30,
    )
    assert table.area is not None and np.ptp(table.area) > 0
    assert not table.omega.flags.writeable
    assert sample_index_table(mode, frequencies, include_area=False).area is None
    with pytest.raises(ModeError, match="distinct"):
        sample_index_table(mode, np.repeat(OMEGA, 4))
    with pytest.raises(ModeError, match="finite"):
        sample_index_table(mode, frequencies, z=float("nan"))


def test_sampled_table_prints_and_rebuilds_on_replace():
    mode = make_provider().modes[0]
    frequencies = np.linspace(0.95 * OMEGA, 1.05 * OMEGA, 9)
    table = sample_index_table(mode, frequencies)
    # The sampler returns an IndexTable subclass that carries the source slopes
    # and the snapshot provenance beside the public table's own surface.
    assert isinstance(table, _SampledIndexTable)
    # The subclass must not regenerate a dataclass repr over its arrays.
    assert "\n" not in repr(table) and "array(" not in repr(table)
    assert repr(table).startswith("IndexTable(")
    assert table.source_identity["area_model"] == "frequency-resolved"
    # A lossless source has no logarithm to interpolate.
    assert table.loss_interpolation == "linear"
    # replace() reconstructs through the subclass constructor, so the Hermite
    # interpolant survives instead of degrading to the base cubic spline.
    without_area = replace(table, area=None)
    assert without_area.area is None
    assert_allclose(
        without_area.dneff_domega(frequencies),
        mode.dneff_domega(frequencies),
        rtol=64 * np.finfo(float).eps,
        atol=1e-30,
    )
    with pytest.raises(ModeError, match="derivative"):
        replace(table, derivative_values=table.derivative_values[:-1])


def test_a_cheap_taper_table_meets_its_held_out_budgets():
    provider, nodes = tapered_provider(n_r=16)
    frequencies = np.linspace(0.98 * OMEGA, 1.02 * OMEGA, 9)
    table = sample_modal_table(provider, frequencies, np.linspace(0, 1, 4), nodes)
    frequency, z = 0.9873 * OMEGA, 0.37
    # One rung of the refinement ladder below, kept in the default lane. The
    # budgets are the defects measured at this rung times a safety factor of
    # three, fixed before the assertion: 1.67e-9 profile, 5.20e-4 1/m generator
    # and 1.27e-5 flux. The flux column is set here by the rule's own Gram error
    # at n_r=16, not by the interpolation.
    errors = validate_modal_table(
        table,
        provider,
        np.array([[frequency, z]]),
        profile_rtol=5e-9,
        generator_atol=1.6e-3,
        flux_atol=4e-5,
    )
    assert errors.shape == (1, 3)
    mode = provider.modes[0]
    exact = (
        mode.neff(np.array([frequency]), z)
        + frequency * mode.dneff_domega(np.array([frequency]), z)
    ) / C
    # Measured 3.98e-9 relative; the same factor of three.
    assert (
        abs(table.generator_derivative(frequency, z)[0, 0] / (-1j * exact[0]) - 1)
        < 1.2e-8
    )


# Budgets fixed before the assertions: the defects measured at 5, 9 and 17
# frequency knots on this taper, each multiplied by a safety factor of three.
# Measured, in the order (profile, generator 1/m, flux, relative beta1):
#    5 knots  3.31e-8,  1.05e-2, 7.00e-7, 3.62e-7
#    9 knots  1.67e-9,  5.20e-4, 7.23e-7, 3.98e-9
#   17 knots  8.63e-13, 2.61e-7, 7.22e-7, 1.01e-10
# The successive ratios are 20 and 1938 for the profile, 20 and 1992 for the
# generator and 91 and 39 for the derivative: this is nowhere near an asymptotic
# h^3 regime, and a "> 4x per rung" order gate would have passed on any of those
# numbers, so the ladder asserts declared absolute budgets instead. The flux
# column does not refine at all, because it is a provider effect rather than an
# interpolation one: the node rule is frozen at the entrance radius, so along a
# taper the panel edge leaves the material interface and the defect grows
# linearly with the displacement.
LADDER = (
    (5, 1.0e-7, 3.2e-2, 2.2e-6, 1.1e-6),
    (9, 5.0e-9, 1.6e-3, 2.2e-6, 1.2e-8),
    (17, 3.0e-12, 8.0e-7, 2.2e-6, 3.1e-10),
)


@pytest.mark.slow
def test_held_out_z_and_derivative_refinement():
    provider, nodes = tapered_provider(n_r=32)
    mode = provider.modes[0]
    frequency, z = 0.9873 * OMEGA, 0.37
    exact = (
        mode.neff(np.array([frequency]), z)
        + frequency * mode.dneff_domega(np.array([frequency]), z)
    ) / C
    tables = []
    for count, profile, generator, flux, derivative in LADDER:
        frequencies = np.linspace(0.98 * OMEGA, 1.02 * OMEGA, count)
        table = sample_modal_table(provider, frequencies, np.linspace(0, 1, 5), nodes)
        tables.append(table)
        validate_modal_table(
            table,
            provider,
            np.array([[frequency, z]]),
            profile_rtol=profile,
            generator_atol=generator,
            flux_atol=flux,
        )
        assert (
            abs(table.generator_derivative(frequency, z)[0, 0] / (-1j * exact[0]) - 1)
            < derivative
        )
    # The frozen node rule, not the knot spacing, sets the flux column on a taper.
    growth = [
        validate_modal_table(
            tables[-1],
            provider,
            np.array([[OMEGA, probe]]),
            profile_rtol=1e-6,
            generator_atol=1.0,
            flux_atol=1e-5,
        )[0, 2]
        for probe in (0.25, 0.5, 1.0)
    ]
    # Linear in the interface displacement: 4.84e-7, 9.68e-7, 1.94e-6 measured,
    # asserted to 1% because they come from the same closed-form displacement.
    assert_allclose(growth, [4.84e-7, 9.68e-7, 1.94e-6], rtol=0.01, atol=0)


def test_a_rotating_degenerate_pair_is_tracked_and_validated():
    base = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = base.node_rule(OMEGA, n_r=16)
    provider = RotatingProvider(base, OMEGA, 20260917)
    frequencies = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 9)
    table = sample_modal_table(
        provider, frequencies, np.array([0.0]), nodes, degenerate_blocks=((0, 1),)
    )
    assert table.degenerate_blocks == ((0, 1),)
    point = np.array([[float(OMEGA), 0.0]])
    # The blocks the table was built with are the blocks it is validated with, so
    # the held-out solve is compared in the subspace it was tracked in: measured
    # 4.60e-16 profile against a declared 1e-12, and 1.24e-5 flux, which is this
    # rule's own Gram error at n_r=16 and not an interpolation defect.
    errors = validate_modal_table(
        table, provider, point, profile_rtol=1e-12, generator_atol=1e-6, flux_atol=4e-5
    )
    assert errors[0, 0] < 1e-12
    # Dropping them leaves the fresh solve rotated by 0.35 rad inside the pair,
    # which per-channel phase alignment cannot undo: measured 0.348 profile error.
    with pytest.raises(ModeError, match="budget"):
        validate_modal_table(
            replace(table, degenerate_blocks=()),
            provider,
            point,
            profile_rtol=1e-12,
            generator_atol=1e-6,
            flux_atol=4e-5,
        )


def test_dense_table_validation_and_optional_reciprocity():
    p = make_provider(radius=250e-9)
    # A deliberately cheap radial rule: it is the tracking and the interpolation
    # that are under test, not the quadrature. Its sampled Gram is 2.41e-4 from
    # the identity at n_r=12 (1.24e-5 at 16, 2.92e-8 at 24), so it declares its
    # own budget instead of the default 1e-4.
    nodes = p.node_rule(OMEGA, n_r=12)
    sample = p.solve(OMEGA, 0.0, nodes=nodes)
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    rows = (tuple(replace(sample, omega=float(w)) for w in omega),)
    table = DenseModalTable(omega, np.array([0.0]), rows, nodes, gram_atol=1e-3)
    for change, message in [
        ({"omega": omega[::-1]}, "increasing"),
        ({"z": np.array([])}, "non-empty"),
        ({"omega": omega[:3]}, "four positive"),
        ({"samples": ((),)}, "sample count"),
        (
            {
                "samples": (
                    tuple(
                        replace(s, gauge_id="other") if i == 0 else s
                        for i, s in enumerate(rows[0])
                    ),
                )
            },
            "disagrees",
        ),
    ]:
        with pytest.raises(ModeError, match=message):
            replace(table, **change)
    no_reciprocal = tuple(
        replace(s, backward_electric=None, backward_magnetic=None, reciprocal_gram=None)
        for s in rows[0]
    )
    bare = replace(table, samples=(no_reciprocal,))
    assert bare.evaluate(OMEGA, 0.0).reciprocal_gram is None
    with pytest.raises(ModeError, match="positive and finite"):
        validate_modal_table(
            table,
            p,
            np.array([[OMEGA, 0.0]]),
            profile_rtol=0.0,
            generator_atol=1.0,
            flux_atol=1.0,
        )


@pytest.mark.parametrize("count", [2, 3])
def test_a_table_refuses_the_z_counts_a_cubic_cannot_carry(count):
    # SciPy's not-a-knot cubic quietly becomes a parabola on three knots and a
    # straight line on two, so such a table would claim an order it does not have.
    p = make_provider(radius=250e-9)
    nodes = p.node_rule(OMEGA, n_r=12)
    sample = p.solve(OMEGA, 0.0, nodes=nodes)
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    z = np.linspace(0.0, 1.0, count)
    rows = tuple(
        tuple(replace(sample, omega=float(w), z=float(position)) for w in omega)
        for position in z
    )
    with pytest.raises(ModeError, match="z knots"):
        DenseModalTable(omega, z, rows, nodes)
    with pytest.raises(ModeError, match="z knots"):
        sample_modal_table(p, omega, z, nodes)


def test_the_cached_interpolants_reproduce_a_direct_reconstruction_exactly():
    # The frequency fit is cached at construction instead of being rebuilt in every
    # evaluate. That is a cost change, not a numerical one: the stacked values, the
    # spline and the order of the two fits are what they were, so the cache must hold
    # exactly what the same reconstruction in this process gives (a structural
    # contract: the cache stores a result, it does not compute another one).
    p = make_provider(radius=250e-9)
    nodes = p.node_rule(OMEGA, n_r=12)
    frequencies = np.linspace(0.98 * OMEGA, 1.02 * OMEGA, 5)
    table = sample_modal_table(p, frequencies, np.array([0.0]), nodes, gram_atol=1e-3)
    omega = 0.9873 * OMEGA
    for name in (
        "electric",
        "magnetic",
        "generator",
        "backward_electric",
        "backward_magnetic",
    ):
        values = np.array([[getattr(s, name) for s in row] for row in table.samples])
        direct = CubicSpline(table.omega, values, axis=1, extrapolate=False)(omega)[0]
        assert (
            table._interpolate(name, omega, 0.0).tobytes()
            == np.asarray(direct).tobytes()
        )


def test_evaluate_reports_the_bracketing_knot_residual():
    p = make_provider(radius=250e-9)
    nodes = p.node_rule(OMEGA, n_r=12)
    frequencies = np.linspace(0.98 * OMEGA, 1.02 * OMEGA, 4)
    table = sample_modal_table(p, frequencies, np.array([0.0]), nodes, gram_atol=1e-3)
    residuals = [s.boundary_residual for s in table.samples[0]]
    middle = float((frequencies[1] + frequencies[2]) / 2)
    np.testing.assert_array_equal(
        table.evaluate(middle, 0.0).boundary_residual,
        np.maximum(residuals[1], residuals[2]),
    )
    # The point's own cell, not the table's first corner.
    assert table.evaluate(middle, 0.0).boundary_residual != residuals[0]
    np.testing.assert_array_equal(
        table.evaluate(float(frequencies[-1]), 0.0).boundary_residual,
        np.maximum(residuals[-2], residuals[-1]),
    )


def test_tracking_refuses_invalid_counts_validity_and_rank():
    p = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = p.node_rule(OMEGA, n_r=12)
    s = p.solve(OMEGA, 0.0, nodes=nodes)
    with pytest.raises(ModeError, match="identical channel"):
        align_modal_sample(
            s,
            s,
            replace(
                nodes,
                xy=nodes.xy[:1],
                weights=nodes.weights[:1],
                region_index=nodes.region_index[:1],
            ),
        )
    with pytest.raises(ModeError, match="physical provider identity"):
        align_modal_sample(s, replace(s, physical_id="different-material"), nodes)
    with pytest.raises(ModeError, match="occupied"):
        align_modal_sample(s, replace(s, valid=np.zeros(2, bool)), nodes)
    with pytest.raises(ModeError, match="unresolved"):
        align_modal_sample(
            s, replace(s, electric=0 * s.electric, magnetic=0 * s.magnetic), nodes
        )
    e = np.repeat(s.electric.sum(axis=2, keepdims=True), 2, axis=2)
    h = np.repeat(s.magnetic.sum(axis=2, keepdims=True), 2, axis=2)
    with pytest.raises(ModeError, match="rank deficient"):
        align_modal_sample(
            s, replace(s, electric=e, magnetic=h), nodes, degenerate_blocks=((0, 1),)
        )


def test_overlap_dominance_separates_a_mistrack_from_a_coarse_knot():
    p = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = p.node_rule(OMEGA, n_r=12)
    reference = p.solve(OMEGA, 0.0, nodes=nodes)
    # Doubling the frequency is far coarser than any table knot spacing and still
    # leaves the weakest aligned overlap at 0.9987. Measured at ratios 1.0, 1.01,
    # 1.2 and 2.0: 0.99999988, 0.99999936, 0.99985 and 0.99872 -- three orders
    # above the floor, which is what lets a dominance criterion be this strict.
    dominance = [
        float(np.min(abs(np.diag(_overlap(reference, other, nodes)))))
        for other in (
            p.solve(ratio * OMEGA, 0.0, nodes=nodes) for ratio in (1.0, 1.01, 1.2, 2.0)
        )
    ]
    assert_allclose(
        dominance, [0.99999988, 0.99999936, 0.99985, 0.99872], rtol=1e-5, atol=0
    )
    assert min(dominance) > 0.99 > MIN_OVERLAP_DOMINANCE

    # A genuine mis-track: each candidate channel is an equal-weight mixture of
    # both reference channels at 0.4 of the amplitude, which is what an
    # unresolved avoided crossing looks like to the assignment. No permutation
    # recovers more than 0.4 of the unit overlap, so tracking must refuse rather
    # than name either of them the partner of a reference channel. The old 1e-8
    # bound accepted this.
    def mix(field):
        return 0.4 * np.stack(
            (field[:, :, 0] + field[:, :, 1], field[:, :, 0] - field[:, :, 1]), axis=-1
        )

    mistracked = replace(
        reference, electric=mix(reference.electric), magnetic=mix(reference.magnetic)
    )
    with pytest.raises(ModeError, match="weakest assigned channel"):
        align_modal_sample(reference, mistracked, nodes)


def test_table_sampling_refuses_invalid_knots_convention_and_kinks(monkeypatch):
    p = make_provider(radius=250e-9)
    nodes = p.node_rule(OMEGA, n_r=12)
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    with pytest.raises(ModeError, match="non-empty z"):
        sample_modal_table(p, omega, np.array([]), nodes)
    monkeypatch.setattr(
        StepIndexProvider, "z_breakpoints", property(lambda self: (0.5,))
    )
    with pytest.raises(ModeError, match="kink"):
        sample_modal_table(p, omega, np.linspace(0.0, 1.0, 4), nodes)
    monkeypatch.setattr(StepIndexProvider, "identity", property(lambda self: {}))
    with pytest.raises(ModeError, match="convention"):
        sample_modal_table(p, omega, np.array([0.0]), nodes)


def test_sampler_frozen_profile_and_invalid_source_derivative(monkeypatch):
    mode = MarcatiliMode(125e-6, "He", 1.0)
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    table = sample_index_table(mode, omega)
    # A frozen transverse profile has exactly the same area at every knot.
    assert np.all(np.asarray(table.area) == mode.aeff())
    monkeypatch.setattr(
        MarcatiliMode,
        "dneff_domega",
        lambda self, omega, z=0.0: np.full_like(omega, np.nan),
    )
    with pytest.raises(ModeError, match="derivative"):
        sample_index_table(mode, omega)


def test_a_table_refuses_a_basis_the_node_rule_has_truncated():
    # The provider normalises over the whole domain and does not renormalise a
    # caller's rule, so a coarse or clipped rule returns a basis that is not
    # orthonormal where it is sampled -- with every sample still `valid`, because
    # nothing was refused. Predeclared: the default budget is 1e-4, the strand's
    # measured knot defect is 2.41e-4 at n_r=12 and below 3e-8 at n_r=24, so the
    # first is refused by name and the second passes with three orders to spare.
    p = make_provider(radius=250e-9)
    frequencies = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    coarse = p.node_rule(OMEGA, n_r=12)
    with pytest.raises(ModeError, match="from orthonormal"):
        sample_modal_table(p, frequencies, np.array([0.0]), coarse)
    fine = p.node_rule(OMEGA, n_r=24)
    table = sample_modal_table(p, frequencies, np.array([0.0]), fine)
    assert 0 < table.gram_defect < 3e-8
    assert table.gram_atol == DEFAULT_GRAM_ATOL
    # The defect the record reports is the worst knot's, and it is the same
    # quantity the refusal measured.
    worst = max(
        float(np.linalg.norm(s.flux_metric - np.eye(1), 2)) for s in table.samples[0]
    )
    assert table.gram_defect == worst
    with pytest.raises(ModeError, match="gram_atol must be positive"):
        sample_modal_table(p, frequencies, np.array([0.0]), fine, gram_atol=0.0)


def grin_provider(count=2, elements=32):
    """Return the reference parabolic sector of the graded-index study."""
    return GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.45 * np.sqrt(0.99)),
        radius=25e-6,
        family="TE",
        count=count,
        elements=elements,
        extent=2.0,
    )


def test_a_provider_channel_table_reads_beta_off_the_tracked_generator():
    # beta is the provider's own generator at the knots, so at a knot the table
    # must reproduce a fresh solve to rounding: 64 epsilon on a propagation
    # constant of about 1e7 1/m, predeclared, covers the spline evaluation and
    # the c/omega round trip through the index.
    provider = grin_provider()
    omega = np.linspace(0.98 * GRIN_OMEGA, 1.02 * GRIN_OMEGA, 9)
    nodes = provider.node_rule(float(np.median(omega)), 0.0)
    table = sample_provider_table(provider, 1, omega, nodes=nodes)
    direct = provider.solve(float(omega[3]), 0.0, nodes=nodes)
    assert_allclose(
        table.beta(omega[3:4]),
        float(np.real(1j * direct.generator[1, 1])),
        rtol=64 * np.finfo(float).eps,
        atol=0,
    )
    identity = dict(cast("_SampledIndexTable", table).source_identity)
    assert identity["channel"] == 1
    assert identity["derivative_model"] == "dense-table-interpolant"
    assert identity["area_model"] == "sampled-vector-surrogate"
    assert float(cast("float", identity["gram_defect"])) < DEFAULT_GRAM_ATOL
    # The stored slopes are the dense interpolant's, not an analytic model's.
    dense = sample_modal_table(provider, omega, np.array([0.0]), nodes)
    expected = [
        float(np.real(1j * dense.generator_derivative(float(w), 0.0)[1, 1]))
        for w in omega
    ]
    assert_allclose(table.dbeta_domega(omega), expected, rtol=0, atol=0)


def test_a_provider_channel_table_refuses_what_it_cannot_sample():
    provider = grin_provider()
    omega = np.linspace(0.98 * GRIN_OMEGA, 1.02 * GRIN_OMEGA, 5)
    with pytest.raises(ModeError, match="outside the 2 tracked channels"):
        sample_provider_table(provider, 2, omega)
    without = cast(
        "_SampledIndexTable",
        sample_provider_table(provider, 0, omega, include_area=False),
    )
    assert without.area is None
    assert dict(without.source_identity)["area_model"] == "none"
    nodes = provider.node_rule(float(np.median(omega)), 0.0)
    positions = np.array([0.0, 0.25, 0.5, 1.0])
    table = sample_modal_table(
        replace(provider, radius=Linear(25e-6, 25.0001e-6, 1.0)),
        omega,
        positions,
        nodes,
    )
    with pytest.raises(ModeError, match="snapshot"):
        _channel_index_table(table, 0, include_area=True)


def _solved_betas(provider, nodes, omega, reference):
    """Return fresh-solve propagation constants in the reference's channel order."""
    sample = align_modal_sample(
        reference, provider.solve(float(omega), 0.0, nodes=nodes), nodes
    )
    return np.real(1j * np.diag(sample.generator))


@pytest.fixture(scope="module")
def held_out_derivatives():
    """Measure held-out beta_1 and beta_2 against fresh-solve five-point stencils.

    The reference parabolic sector on the nine-knot +-2 % band a run would use.
    ``beta_1`` and ``beta_2`` come from the consumers' route in ``modes.base``,
    and the reference for both is a five-point central difference of *fresh*
    provider solves at a relative step of 1e-3: small enough that a five-point
    stencil's truncation on a function smooth on the scale of omega is 1e-12
    relative, and large enough that the rounding of a 1e7 1/m propagation
    constant leaves the second derivative resolved to about 1e-3 (measured, by
    repeating the stencil at twice the step, in a development study of the
    provider-table derivatives).
    """
    provider = grin_provider()
    omega = np.linspace(0.98 * GRIN_OMEGA, 1.02 * GRIN_OMEGA, 9)
    nodes = provider.node_rule(float(np.median(omega)), 0.0)
    dense = sample_modal_table(provider, omega, np.array([0.0]), nodes)
    modes = ModalGuide.from_provider(provider, omega, nodes=nodes).mode_collection.modes
    worst = {"beta1": 0.0, "beta2": 0.0}
    for held in (omega[:-1] + omega[1:]) / 2:
        point = np.array([float(held)])
        anchor = align_modal_sample(
            dense.evaluate(float(held), 0.0),
            provider.solve(float(held), 0.0, nodes=nodes),
            nodes,
        )
        step = 1e-3 * float(held)
        solved = {
            offset: _solved_betas(provider, nodes, held + offset * step, anchor)
            for offset in (-2, -1, 1, 2)
        }
        centre = np.real(1j * np.diag(anchor.generator))
        first = (solved[-2] - 8 * solved[-1] + 8 * solved[1] - solved[2]) / (12 * step)
        second = (
            -solved[-2] + 16 * solved[-1] - 30 * centre + 16 * solved[1] - solved[2]
        ) / (12 * step**2)
        for channel, mode in enumerate(modes):
            worst["beta1"] = max(
                worst["beta1"],
                abs(float(beta1(mode, point)[0]) - first[channel])
                / abs(first[channel]),
            )
            try:
                measured = float(beta2(mode, point)[0])
            except ModeError:
                # beta2 differences beta1 0.6 % either side of the point and the
                # table refuses to extrapolate, so the outermost held-out
                # frequencies of a narrow band have no second derivative at all.
                continue
            worst["beta2"] = max(
                worst["beta2"],
                abs(measured - second[channel]) / abs(second[channel]),
            )
    return worst


def test_a_provider_channel_reproduces_its_first_derivative_held_out(
    held_out_derivatives,
):
    # beta_1 is the interpolant's own and the guide certifies nothing about it
    # (from_provider checks beta alone), so it is measured here. Predeclared 5e-12:
    # a factor of ten on the 4.4e-13 that
    # the provider-table derivative study measures on this knot
    # set, whose own floor is the stencil's rounding, about 3e-13.
    assert held_out_derivatives["beta1"] < 5e-12


def test_a_provider_channel_second_derivative_is_only_a_diagnostic(
    held_out_derivatives,
):
    # This records the order beta_2 reaches; it does not certify it. Predeclared
    # 1e-1 on the 2.75e-2 measured on this knot set. The second assertion is the
    # one that matters: the same interpolant that gives beta_1 to 1e-13 gives
    # beta_2 to one part in forty, because a cubic through a nearly constant group
    # delay carries almost no curvature. Consumers are modes.base.beta2, zdw,
    # tools.dispersion_length and tools.soliton_order.
    assert held_out_derivatives["beta2"] < 1e-1
    assert held_out_derivatives["beta2"] > 1e3 * held_out_derivatives["beta1"]


def _declared_sample(rule, field):
    """Return a one-channel sample carrying a declared transverse x field."""
    nodes = len(rule.weights)
    electric = np.zeros((nodes, 3, 1), dtype=complex)
    electric[:, 0, 0] = field
    return ModalSample(
        1.0e15,
        0.0,
        electric,
        np.zeros((nodes, 3, 1), dtype=complex),
        np.array([[-1j * 1.0e7]]),
        np.array([[1.0 + 0j]]),
        np.zeros(1),
        np.ones(1, dtype=bool),
        None,
        None,
        None,
        "declared-field",
        "declared-field",
    )


def test_the_surrogate_area_of_a_channel_is_its_quartic_overlap():
    # Two fields whose A_eff is known in closed form, rather than the production
    # expression typed a second time: a uniform field on a disc has the area of
    # the disc, and a Gaussian of 1/e field radius w has pi w^2.
    radius = 3e-6
    disc = radial_node_rule((), radius, n_r=8, n_theta=4)
    uniform = _declared_sample(disc, np.ones(len(disc.weights)))
    # Exact: the polar Jacobian of a constant is a polynomial the Gauss rule
    # integrates exactly, so only the rounding of two sums and a division enters.
    assert_allclose(
        channel_area(uniform, disc, 0),
        np.pi * radius**2,
        rtol=64 * np.finfo(float).eps,
        atol=0,
    )
    width = 2e-6
    # Out to six field radii the discarded tail of |E|^2 is exp(-72), so the bound
    # is the radial Gauss rule's own error on a smooth non-polynomial. Predeclared
    # 1e-12 at 64 nodes; measured 8e-15.
    rule = radial_node_rule((), 6 * width, n_r=64, n_theta=4)
    distance = np.hypot(rule.xy[:, 0], rule.xy[:, 1])
    gaussian = _declared_sample(rule, np.exp(-((distance / width) ** 2)))
    assert_allclose(
        channel_area(gaussian, rule, 0), np.pi * width**2, rtol=1e-12, atol=0
    )


class CrossingProvider:
    """Give a provider's two physical channels propagation constants that cross.

    The fields stay the base provider's, two distinct physical channels; only the
    generator is replaced by two straight lines that exchange order at the middle
    of the band. It is the situation an avoided crossing puts a tracker in: the
    eigenvalue order is no longer the channel order, and only the physical overlap
    says which channel is which.
    """

    def __init__(self, base, centre, beta, slope, loss=0.0):
        self._base, self._centre = base, centre
        self._beta, self._slope, self._loss = beta, slope, loss

    def betas(self, omega):
        """Return the two declared propagation constants at one frequency.

        ``loss`` is the amplitude attenuation in 1/m: the generator is
        ``-i beta``, so ``- i loss`` here becomes ``Re G = -loss``, which is loss
        in the internal convention and gain when it is negative.
        """
        detuning = self._slope * (omega / self._centre - 1)
        return (
            np.array([self._beta + detuning, self._beta - detuning], dtype=complex)
            - 1j * self._loss
        )

    @property
    def regions(self):
        return self._base.regions

    @property
    def z_breakpoints(self):
        return self._base.z_breakpoints

    @property
    def z_domain(self):
        return self._base.z_domain

    @property
    def omega_domain(self):
        return self._base.omega_domain

    @property
    def identity(self):
        return self._base.identity

    def node_rule(self, omega, z=0.0, *, n_r=None, n_theta=None):
        return self._base.node_rule(omega, z, n_r=n_r, n_theta=n_theta)

    def solve(self, omega, z, *, nodes):
        sample = self._base.solve(omega, z, nodes=nodes)
        return replace(sample, generator=-1j * np.diag(self.betas(omega)))


def test_tracking_follows_the_physical_channel_through_an_avoided_crossing():
    base = make_provider(descriptors=(("HE", 1, 1, "even"), ("TE", 0, 1, "even")))
    omega = np.linspace(0.98 * OMEGA, 1.02 * OMEGA, 9)
    nodes = base.node_rule(OMEGA, n_r=24)
    provider = CrossingProvider(base, OMEGA, 1.0e7, 5.0e5)
    table = sample_modal_table(provider, omega, np.array([0.0]), nodes)
    tracked = np.array(
        [np.real(1j * np.diag(sample.generator)) for sample in table.samples[0]]
    )
    declared = np.array([provider.betas(float(w)) for w in omega])
    # Exactly the declared branches, knot by knot: the assignment is by physical
    # overlap of the fields, which the crossing does not touch, so no interpolation
    # or tolerance is involved.
    np.testing.assert_array_equal(tracked, declared)
    # Sorting by eigenvalue would have swapped the two channels at the far end of
    # the band, which is what this test exists to refuse.
    assert tracked[0, 0] < tracked[0, 1] and tracked[-1, 0] > tracked[-1, 1]
    assert not np.array_equal(tracked, np.sort(declared, axis=1))
    # The bridge reads the same branch, and each channel keeps its own field.
    first = _channel_index_table(table, 0, include_area=True)
    np.testing.assert_array_equal(first.beta(omega), declared[:, 0])
    assert first.effective_area(omega[:1]) != _channel_index_table(
        table, 1, include_area=True
    ).effective_area(omega[:1])


def test_a_channel_table_refuses_a_generator_that_declares_gain():
    # A provider's generator is internal-convention, G = -i (omega/c) conj(n_eff),
    # so a passive channel has Re G <= 0 and a positive real part is gain. No
    # provider in the package produces one, so the refusal needs a declared
    # generator. This reading used to be Im beta >= 0, which had the sign backwards.
    base = make_provider(descriptors=(("HE", 1, 1, "even"), ("TE", 0, 1, "even")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    nodes = base.node_rule(OMEGA, n_r=24)
    table = sample_modal_table(
        CrossingProvider(base, OMEGA, 1.0e7, 0.0, loss=-1.0),
        omega,
        np.array([0.0]),
        nodes,
    )
    assert np.real(table.samples[0][0].generator[0, 0]) > 0
    with pytest.raises(ModeError, match="declares gain"):
        _channel_index_table(table, 0, include_area=False)


def test_a_channel_table_refuses_a_generator_that_couples_its_channels():
    # A channel's propagation constant is the generator's diagonal, so a coupled
    # generator has none: its flow is a matrix exponential, which belongs to the
    # coupled-flow leaf of a later phase. Measured off-diagonal norm of every
    # provider here: exactly zero.
    base = make_provider(descriptors=(("HE", 1, 1, "even"), ("TE", 0, 1, "even")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    nodes = base.node_rule(OMEGA, n_r=24)
    table = sample_modal_table(base, omega, np.array([0.0]), nodes)
    assert np.all(
        table.samples[0][0].generator == np.diag(np.diag(table.samples[0][0].generator))
    )
    coupled = tuple(
        tuple(
            replace(sample, generator=sample.generator + np.array([[0, 1j], [1j, 0]]))
            for sample in row
        )
        for row in table.samples
    )
    with pytest.raises(ModeError, match="diagonal generator"):
        _channel_index_table(replace_table(table, coupled), 0, include_area=False)


def test_a_channel_table_carries_loss_from_the_generator_to_the_operator():
    # The provider's generator is internal-convention, G = -i (omega/c) conj(n_eff),
    # so Re G = -alpha/2 and a passive channel has Re G < 0. The P-layer IndexTable
    # is optical, where Im n_eff > 0 is loss, so the bridge conjugates once and the
    # table-to-operator site conjugates back: the ordinary two-site round trip of
    # any optical table, not a fourth conjugation site.
    base = make_provider(radius=250e-9)
    nodes = base.node_rule(OMEGA, n_r=24)
    reference = base.solve(OMEGA, 0.0, nodes=nodes)
    real_index = float(np.real(1j * reference.generator[0, 0])) * C / OMEGA
    # A constant imaginary index, sized for an amplitude rate of 0.5 1/m at OMEGA.
    # Constant so that the table's linear loss interpolation is exact between the
    # knots and the comparisons below are rounding, not interpolation.
    imaginary_index = 0.5 * C / OMEGA
    # A synthetic band wide enough to cover a grid's whole active mask; only the
    # generator is declared, and the fields stay the solved channel's.
    omega = np.linspace(0.4 * OMEGA, 3.0 * OMEGA, 8)
    samples = tuple(
        replace(
            reference,
            omega=float(frequency),
            generator=np.array(
                [[-frequency / C * (imaginary_index + 1j * real_index)]]
            ),
        )
        for frequency in omega
    )
    table = _channel_index_table(
        DenseModalTable(omega, np.array([0.0]), (samples,), nodes, ()),
        0,
        include_area=False,
    )
    probe = np.array([OMEGA])
    # Optical convention on the way out: Im n_eff is positive and is exactly the
    # index the internal real part declared. 16 epsilon covers the multiply,
    # the divide and the interpolant's evaluation.
    assert_allclose(
        np.imag(table.neff(probe)),
        imaginary_index,
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )
    assert_allclose(
        np.real(table.neff(probe)),
        real_index,
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )
    mode = ProviderChannelMode(
        table, ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even"), 1e-6, 1e-12
    )
    assert_allclose(
        alpha(mode, probe),
        2 * OMEGA * imaginary_index / C,
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )
    # And through the operator route the amplitude decays: exp(D z) is the
    # amplitude, so Re D = -alpha/2 everywhere in the active band.
    grid = EnvelopeGrid.from_limits(1.0, 835e-9, (600e-9, 1000e-9), 2e-12)
    active = grid.active_mask
    operator = dispersion_operator(mode, grid, Frame.comoving(grid))[active]
    assert np.all(np.real(operator) < 0)
    assert np.all(np.exp(np.real(operator)) < 1)
    assert_allclose(
        np.real(operator),
        -grid.omega[active] * imaginary_index / C,
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )


def test_a_channel_table_measures_coupling_against_the_channel_splitting():
    # Two channels split by 1 1/m and coupled by 1e-3 1/m mix completely over a few
    # metres, so they have no per-channel propagation constant. The old metric
    # compared the off-diagonal with |beta| ~ 1e7 1/m and admitted it; the
    # splitting is the physical scale and refuses it.
    base = make_provider(descriptors=(("HE", 1, 1, "even"), ("TE", 0, 1, "even")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    nodes = base.node_rule(OMEGA, n_r=24)
    near = CrossingProvider(base, OMEGA, 1.0e7, 0.0)
    table = sample_modal_table(near, omega, np.array([0.0]), nodes)
    coupled = tuple(
        tuple(
            replace(
                sample,
                generator=np.array(
                    [[-1j * 1.0e7, -1j * 1e-3], [-1j * 1e-3, -1j - 1j * 1.0e7]]
                ),
            )
            for sample in row
        )
        for row in table.samples
    )
    record = replace_table(table, coupled)
    off_diagonal = float(
        np.linalg.norm(
            record.samples[0][0].generator
            - np.diag(np.diag(record.samples[0][0].generator))
        )
    )
    # What the old metric saw: the off-diagonal against the norm of the diagonal.
    old_scale = float(np.linalg.norm(np.diag(record.samples[0][0].generator)))
    assert off_diagonal < MAX_GENERATOR_COUPLING * old_scale
    with pytest.raises(ModeError, match="smallest splitting"):
        _channel_index_table(record, 0, include_area=False)


def replace_table(table, samples):
    """Return the same table with different samples, keeping its knots and rule."""
    return DenseModalTable(
        table.omega, table.z, samples, table.nodes, table.degenerate_blocks
    )


def test_cutoff_policy():
    # The policy is: an occupied band a channel cannot support is refused when the
    # table is built, by name, not masked or extrapolated into. TE01 sized for
    # V = 2.5 at 835 nm is bound there (w = 0.358) and falls to w = 0.0635 at
    # 866 nm, below the declared PROFILE_MIN_W of 0.1.
    wavelength = 835e-9
    radius = 2.5 / (2 * np.pi / wavelength * np.sqrt(1.45**2 - 1.0))
    mode = StepIndexMode(
        radius,
        core=1.45,
        cladding=1.0,
        profile_wavelength=wavelength,
        kind="TE",
        n=0,
        m=1,
    )
    provider = ModalGuide.from_step_index(
        [mode], core=IndexMaterial(1.45), cladding=IndexMaterial(1.0)
    ).field_provider
    bound = 2 * np.pi * C / np.linspace(800e-9, 840e-9, 4)[::-1]
    nodes = provider.node_rule(float(np.median(bound)), n_r=24)
    table = sample_modal_table(provider, bound, np.array([0.0]), nodes)
    assert np.all(table.samples[0][0].valid)
    # Reaching past the knots is refused rather than extrapolated: a band the
    # table does not carry is not a band it may guess at.
    with pytest.raises(ModeError, match="does not extrapolate"):
        table.evaluate(float(2 * np.pi * C / 866e-9), 0.0)
    # Two refusals, both by name: a channel the frozen profile cannot resolve,
    # and one the guide does not support at all.
    near = 2 * np.pi * C / np.linspace(820e-9, 866e-9, 4)[::-1]
    with pytest.raises(ModeError, match="too close to cutoff"):
        sample_modal_table(provider, near, np.array([0.0]), nodes)
    below = 2 * np.pi * C / np.linspace(820e-9, 870e-9, 4)[::-1]
    with pytest.raises(ModeError, match="TE01 is below cutoff"):
        sample_modal_table(provider, below, np.array([0.0]), nodes)


# ------------------------------------------- the declared change of basis


def circular_table(provider, omega, nodes, **changes):
    """Build one snapshot table of a parity pair in the declared circular basis."""
    basis = circular_pair(provider_channels(provider), (0, 1))
    fields = {"degenerate_blocks": ((0, 1),), "basis": basis} | changes
    return basis, sample_modal_table(provider, omega, np.array([0.0]), nodes, **fields)


def test_a_declared_basis_transforms_the_reference_and_composes_its_gauge():
    """`e' = e U`, `W' = U.H W U`, and the gauge is `<tracking>+<basis>`."""
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    nodes = provider.node_rule(0.99 * OMEGA, n_r=12)
    plain = sample_modal_table(
        provider, omega, np.array([0.0]), nodes, degenerate_blocks=((0, 1),)
    )
    basis, turned = circular_table(provider, omega, nodes)
    matrix = basis.transform
    for before, after in zip(plain.samples[0], turned.samples[0], strict=True):
        # `max|E|` is 1.55e7, so an atol of 1e-12 was an exact-equality test with
        # the wrong name on it. The transform is one product with a 2 x 2 unitary
        # matrix: two complex terms per element, a few roundings each, held normwise
        # to 4 eps.
        assert_within(after.electric, before.electric @ matrix, 4 * EPS, what="E")
        assert_within(after.magnetic, before.magnetic @ matrix, 4 * EPS, what="H")
        assert_allclose(
            after.flux_metric,
            matrix.conj().T @ before.flux_metric @ matrix,
            rtol=0,
            atol=8 * np.finfo(float).eps,
        )
        # The generator of an exactly degenerate block is invariant, and it is
        # carried over rather than multiplied out, so it is exactly unchanged.
        assert np.array_equal(after.generator, before.generator)
        assert after.gauge_id == f"{before.gauge_id}+circular(0,1)"
    assert turned.basis is basis
    assert composed_gauge("g", None) == "g"
    # The declared basis leaves the sampled Gram defect exactly where it was: the
    # spectral norm is unitarily invariant.
    assert turned.gram_defect == pytest.approx(plain.gram_defect, rel=1e-12)


def test_a_declared_basis_puts_a_held_out_solve_in_the_same_coordinates():
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 5)
    nodes = provider.node_rule(0.99 * OMEGA, n_r=12)
    basis, table = circular_table(provider, omega, nodes)
    middle = float((omega[1] + omega[2]) / 2)
    fresh = solved_in_table_basis(table, provider, middle, 0.0)
    direct = apply_channel_basis(provider.solve(middle, 0.0, nodes=nodes), basis)
    # The same solve and the same basis product reached through the table's helper.
    assert_within(fresh.electric, direct.electric, 4 * EPS, what="E")
    errors = validate_modal_table(
        table,
        provider,
        np.array([[middle, 0.0]]),
        profile_rtol=1e-3,
        generator_atol=1.0,
        flux_atol=1e-3,
    )
    assert errors.shape == (1, 3)


def test_a_basis_over_an_undeclared_block_is_refused_by_name():
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4)
    nodes = provider.node_rule(0.99 * OMEGA, n_r=12)
    with pytest.raises(ModeError, match="do not name"):
        circular_table(provider, omega, nodes, degenerate_blocks=())


def test_a_basis_of_the_wrong_width_is_refused_by_name():
    provider = make_provider()
    nodes = provider.node_rule(OMEGA, n_r=12)
    sample = provider.solve(OMEGA, 0.0, nodes=nodes)
    labels = (
        ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even"),
        ChannelLabel("HE", 1, 1, "odd", None, "HE1,1:odd"),
    )
    with pytest.raises(ModeError, match="describes 2 channel"):
        apply_channel_basis(sample, circular_pair(labels, (0, 1)))


def test_a_split_block_has_its_product_evaluated_rather_than_refused():
    """A split block is a coupled flow, not an error.

    For a declared basis whose block's propagation constants differ,
    ``U.H diag(beta) U`` is not diagonal; the general route executes it as the
    ``linear.coupled_flow`` leaf, so the split case is **admitted here** and the product
    is evaluated rather than the generator carried over; the route builds the flow from
    it (:func:`solprop.translate.general.coupled_block`).
    """
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = provider.node_rule(OMEGA, n_r=12)
    sample = provider.solve(OMEGA, 0.0, nodes=nodes)
    basis = circular_pair(provider_channels(provider), (0, 1))
    generator = np.array(sample.generator, dtype=complex)
    generator[1, 1] += -1j * 1e-3 * abs(generator[1, 1])
    split = replace(sample, generator=generator)
    turned = apply_channel_basis(split, basis)
    matrix = basis.transform
    assert_allclose(
        turned.generator,
        matrix.conj().T @ generator @ matrix,
        rtol=0,
        atol=64 * np.finfo(float).eps * float(np.max(abs(generator))),
    )
    diagonal = np.diag(np.diag(turned.generator))
    assert float(np.max(abs(turned.generator - diagonal))) > 0.0
    # And the exactly degenerate case is untouched: the generator is carried over
    # as a copy, which is what keeps the degenerate case's numbers where they were.
    degenerate = apply_channel_basis(sample, basis)
    assert np.array_equal(degenerate.generator, sample.generator)


def test_an_evaluated_split_block_that_shows_gain_is_refused_by_name():
    """The passivity gate the carried-over path gets for free."""
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = provider.node_rule(OMEGA, n_r=12)
    sample = provider.solve(OMEGA, 0.0, nodes=nodes)
    labels = provider_channels(provider)
    basis = circular_pair(labels, (0, 1))
    generator = np.array(sample.generator, dtype=complex)
    generator[1, 1] += -1j * 1e-3 * abs(generator[1, 1])
    # A transform that is unitary but is not the one the sample is turned by would
    # be caught by the invariance gate; this injects the gain directly, which is
    # what a conjugation slipped into the product looks like.
    broken = replace(sample, generator=generator)
    turned = apply_channel_basis(broken, basis)
    active = np.array(turned.generator, dtype=complex)
    active[0, 0] += 1.0
    with pytest.raises(ModeError, match="cannot make a passive block active"):
        _check_transformed_passivity(broken, replace(turned, generator=active), basis)


def test_a_coupled_generator_is_refused_before_the_degeneracy_gate():
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = provider.node_rule(OMEGA, n_r=12)
    sample = provider.solve(OMEGA, 0.0, nodes=nodes)
    basis = circular_pair(provider_channels(provider), (0, 1))
    generator = np.array(sample.generator, dtype=complex)
    generator[0, 1] = -1j * 1e-3 * abs(generator[0, 0])
    with pytest.raises(ModeError, match="already diagonal"):
        apply_channel_basis(replace(sample, generator=generator), basis)


def test_a_transform_that_changes_the_gram_defect_is_refused_by_name():
    """The spectral norm is unitarily invariant, so a change means nonunitary.

    The declaration's own budget catches a nonunitary matrix at construction, so
    this gate is reached by pushing one past it: the transform applied here is the
    one the record checked, scaled, which is what a caller building a record by
    hand or a future non-circular constructor could produce.
    """
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = provider.node_rule(OMEGA, n_r=12)
    sample = provider.solve(OMEGA, 0.0, nodes=nodes)
    basis = circular_pair(provider_channels(provider), (0, 1))
    object.__setattr__(basis, "transform", 1.5 * basis.transform)
    with pytest.raises(ModeError, match="unitarily invariant"):
        apply_channel_basis(sample, basis)


def scaled_pair(labels, scale):
    """Return the circular pair's declaration with its matrix scaled by `scale`.

    ``U.H U = scale^2 I``, so the declared unitary defect is exactly
    ``scale^2 - 1`` and the transformed metric is ``scale^2`` times the
    reference's: the measured norm moves by the **unitary defect** and not by the
    metric's own departure from the identity.
    """
    circular = circular_pair(labels, (0, 1))
    return ChannelBasis(
        circular.transform * scale, ((0, 1),), circular.labels, circular.gauge_id
    )


def test_the_invariance_gate_admits_what_the_declaration_s_budget_admits():
    """The two gates agree: `BASIS_UNITARY_ATOL` is the floor of the per-knot one.

    A per-knot gate at the rounding of the product alone -- 2.8e-14 for two
    channels -- would refuse a declaration the record has just certified at 1e-12,
    because the norm it measures moves by the transform's own unitary defect. So
    the floor is the declaration's budget, and a defect inside it passes both.
    """
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    nodes = provider.node_rule(OMEGA, n_r=12)
    sample = provider.solve(OMEGA, 0.0, nodes=nodes)
    labels = provider_channels(provider)
    inside = 0.2 * BASIS_UNITARY_ATOL
    admitted = scaled_pair(labels, float(np.sqrt(1.0 + inside)))
    # The declaration admits it, at the defect asked for -- to a part in a
    # thousand, which is the rounding of a `sqrt` and a square at this size.
    assert_allclose(admitted.unitary_defect, inside, rtol=1e-2, atol=0.0)
    turned = apply_channel_basis(sample, admitted)
    # And so does the per-knot gate, although the norm did move by that defect --
    # well above the rounding of the product, which is what the floor is for.
    before = float(np.linalg.norm(sample.flux_metric - np.eye(2), ord=2))
    after = float(np.linalg.norm(turned.flux_metric - np.eye(2), ord=2))
    assert abs(after - before) > 64 * 2 * float(np.finfo(float).eps)
    assert abs(after - before) <= BASIS_UNITARY_ATOL
    # An order above the budget fails the first gate, where it belongs.
    with pytest.raises(ModeError, match="must be unitary"):
        scaled_pair(labels, float(np.sqrt(1.0 + 2.0 * BASIS_UNITARY_ATOL)))

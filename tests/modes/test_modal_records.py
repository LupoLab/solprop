# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise production validation independently of the session's shape checker."""

from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose
from witness_materials import absorbing_silica, malitson_silica

import solprop.modes._stepindex_vector as vector_module
import solprop.modes.guide as guide_module
from modal_provider_cases import OMEGA, make_provider
from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.modes import ModeCollection, StepIndexMode
from solprop.modes.graded_index import GradedIndexProvider, PowerLawPermittivity
from solprop.modes.guide import (
    ChannelBasis,
    ModalGuide,
    ModalSampling,
    ProviderChannelMode,
    circular_pair,
    provider_channels,
)
from solprop.modes.material import IndexMaterial, MaterialKerr, MaterialRegion
from solprop.modes.protocol import ChannelLabel
from solprop.modes.tabulated import _SampledIndexTable
from solprop.modes.vector import ModalNodeRule, ModalProvider
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.physdata import fused_silica
from solprop.polarisation import stokes
from solprop.profiles import Linear
from solprop.translate.modal_tables import channel_area


@pytest.fixture(scope="module")
def sample():
    provider = make_provider(radius=250e-9)
    nodes = provider.node_rule(OMEGA, n_r=24)
    return provider.solve(OMEGA, 0.0, nodes=nodes)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("omega", 0.0, "omega"),
        ("z", float("nan"), "finite"),
        ("gauge_id", "", "non-empty"),
        ("physical_id", "", "non-empty"),
        ("boundary_residual", np.array([-1.0]), "non-negative"),
        ("magnetic", np.zeros((2, 3, 1), complex), "shape"),
        ("generator", np.ones((2, 2), complex), "shape"),
        ("flux_metric", np.array([[-1.0 + 0j]]), "positive definite"),
        ("flux_metric", np.array([[1.0 + 1j]]), "Hermitian"),
        ("backward_electric", None, "together"),
        ("reciprocal_gram", np.zeros((1, 1), complex), "nonsingular"),
        ("electric", np.full((1, 3, 1), complex("nan")), "finite"),
    ],
)
def test_sample_validation(sample, field, value, message):
    with pytest.raises(ModeError, match=message):
        replace(sample, **{field: value})


def test_record_copy_ownership(sample):
    bad = replace(sample)
    object.__setattr__(bad, "valid", np.array([float("nan")]))
    with pytest.raises(ModeError, match="boolean mask"):
        bad.__post_init__()
    assert not sample.electric.flags.writeable
    copy = sample.electric.copy()
    owned = replace(sample, electric=copy)
    copy[:] = 0
    assert np.any(owned.electric)
    assert (
        replace(
            sample, backward_electric=None, backward_magnetic=None, reciprocal_gram=None
        ).reciprocal_gram
        is None
    )


@pytest.mark.parametrize(
    "xy,weights,regions,message",
    [
        (np.empty((0, 2)), np.empty(0), np.empty(0, np.int32), "at least one"),
        (np.zeros((2, 2)), np.ones(1), np.zeros(1, np.int32), "agree"),
        (np.full((1, 2), np.nan), np.ones(1), np.zeros(1, np.int32), "finite"),
        (np.zeros((1, 2)), np.zeros(1), np.zeros(1, np.int32), "positive"),
        (np.zeros((1, 2)), np.ones(1), np.array([-1], np.int32), "non-negative"),
    ],
)
def test_node_validation(xy, weights, regions, message):
    with pytest.raises(ModeError, match=message):
        ModalNodeRule(xy, weights, regions, {})


def test_material_validation():
    with pytest.raises(ModeError, match="index"):
        IndexMaterial(-1.0)
    material = IndexMaterial(1.45)
    assert material.permittivity([OMEGA]).shape == (1,)
    assert material.identity["index"] == 1.45
    with pytest.raises(ModeError, match="frequencies"):
        material.permittivity([0.0])
    with pytest.raises(ModeError, match="region"):
        MaterialRegion("", 0.0, 1.0, material)


def test_a_region_declares_one_kerr_response_or_is_linear():
    """A region owns its response; two of a kind, or a stranger, is refused."""
    material = IndexMaterial(1.45)
    assert MaterialRegion("clad", 1.0, None, material).kerr is None
    kerr = MaterialKerr(chi3=2.2e-22)
    assert MaterialRegion("core", 0.0, 1.0, material, (kerr,)).kerr is kerr
    with pytest.raises(ModeError, match="one material has one"):
        MaterialRegion("core", 0.0, 1.0, material, (kerr, MaterialKerr(n2=1e-20)))
    # Past the constructor, as the session's runtime hook rejects the annotation
    # before `__post_init__` is reached; in production no hook runs and this check
    # is the only one there is.
    stranger = MaterialRegion("core", 0.0, 1.0, material, (kerr,))
    object.__setattr__(stranger, "responses", (object(),))
    with pytest.raises(ModeError, match="at most one of each"):
        stranger.__post_init__()
    # The delayed tensor response joins the union, and is a share of the region's
    # own chi3, so it needs one beside it.
    raman = SilicaTensorResponse()
    assert MaterialRegion("core", 0.0, 1.0, material, (kerr, raman)).raman is raman
    assert MaterialRegion("core", 0.0, 1.0, material, (kerr,)).raman is None
    with pytest.raises(ModeError, match="more than one SilicaTensorResponse"):
        MaterialRegion("core", 0.0, 1.0, material, (kerr, raman, raman))
    with pytest.raises(ModeError, match="no MaterialKerr"):
        MaterialRegion("core", 0.0, 1.0, material, (raman,))


def test_a_material_kerr_response_converts_at_the_frequency_it_is_asked_for():
    """``chi3 = 4 eps0 c n0^2 n2 / 3``, with ``n0`` the region's own index there.

    A declared ``chi3`` is used as it stands; a declared ``n2`` is converted
    against the material's linear index at the frequency supplied, and the round
    trip through ``nonlinear_index`` returns what was declared.
    """
    silica = IndexMaterial(fused_silica())
    declared = MaterialKerr(n2=2.6e-20)
    index = float(np.sqrt(np.real(silica.permittivity([OMEGA])[0])))
    assert_allclose(
        declared.susceptibility(silica, OMEGA),
        4 * 2.6e-20 * EPS0 * C * index**2 / 3,
        rtol=1e-15,
        atol=0.0,
    )
    assert_allclose(
        declared.nonlinear_index(silica, OMEGA), 2.6e-20, rtol=1e-15, atol=0.0
    )
    absolute = MaterialKerr(chi3=2.2e-22)
    assert absolute.susceptibility(silica, OMEGA) == 2.2e-22
    assert_allclose(
        absolute.nonlinear_index(silica, OMEGA),
        3 * 2.2e-22 / (4 * EPS0 * C * index**2),
        rtol=1e-15,
        atol=0.0,
    )
    # The conversion is the region's own index, so it moves with the frequency.
    assert declared.susceptibility(silica, 1.2 * OMEGA) != declared.susceptibility(
        silica, OMEGA
    )


def test_an_n2_declares_the_wavelength_it_was_measured_at():
    """`at=` fixes where the linear index of the conversion is read.

    An n2 is a measured number and the measurement has a wavelength. The same n2
    declared at 800 nm and at 2 um gives two chi3 values, and their ratio is the square
    of the index ratio between those wavelengths and nothing else -- which is what makes
    the declaration necessary rather than decorative, since the leaf carries one number.
    """
    silica = IndexMaterial(fused_silica())
    blue, red = 800e-9, 2e-6
    at_blue = MaterialKerr(n2=2.6e-20, at=blue)
    at_red = MaterialKerr(n2=2.6e-20, at=red)
    # Asked at any frequency, each converts at its own declared wavelength.
    for omega in (OMEGA, 1.3 * OMEGA):
        assert at_blue.susceptibility(silica, omega) == at_blue.susceptibility(
            silica, OMEGA
        )
    indices = [
        float(np.sqrt(np.real(silica.permittivity([2 * np.pi * C / one])[0])))
        for one in (blue, red)
    ]
    ratio = at_blue.susceptibility(silica, OMEGA) / at_red.susceptibility(silica, OMEGA)
    assert_allclose(ratio, (indices[0] / indices[1]) ** 2, rtol=1e-14, atol=0.0)
    # Measured: 2.1302 % between 800 nm and 2 um on optmat's silica (2.1296 % on the
    # registry's before SEP 3's Phase 4a), above the route's 1 % budget.
    assert ratio - 1.0 == pytest.approx(0.0213017, rel=1e-4)
    # An undeclared n2 still converts where it is asked, which is the default.
    assert MaterialKerr(n2=2.6e-20).susceptibility(
        silica, 2 * np.pi * C / blue
    ) == at_blue.susceptibility(silica, OMEGA)
    # A declared n2 is returned as it stands, whatever it was measured at.
    assert at_red.nonlinear_index(silica, OMEGA) == 2.6e-20


def test_a_declared_measurement_wavelength_outside_the_index_model_is_named():
    """The refusal says it was the declaration, not the frequency asked for."""
    silica = IndexMaterial(fused_silica())
    with pytest.raises(ModeError, match="declares its n2 measured at"):
        MaterialKerr(n2=2.6e-20, at=1e-9).susceptibility(silica, OMEGA)
    # Without a declaration the generic refusal stands, naming the domain.
    with pytest.raises(ModeError, match="outside the material domain"):
        MaterialKerr(n2=2.6e-20).susceptibility(silica, 2 * np.pi * C / 1e-9)


@pytest.mark.parametrize(
    "arguments,message",
    [
        ({}, "exactly one"),
        ({"chi3": 1e-22, "n2": 1e-20}, "exactly one"),
        ({"chi3": -1e-22}, "non-negative"),
        ({"n2": float("inf")}, "non-negative"),
        ({"chi3": 1e-22, "at": 800e-9}, "declares chi3"),
        ({"n2": 1e-20, "at": 0.0}, "wavelength in metres"),
        ({"n2": 1e-20, "at": -1.0}, "wavelength in metres"),
        ({"n2": 1e-20, "at": float("nan")}, "wavelength in metres"),
    ],
)
def test_a_material_kerr_response_refuses_what_it_cannot_carry(arguments, message):
    with pytest.raises(ModeError, match=message):
        MaterialKerr(**arguments)


@pytest.mark.parametrize("omega", [0.0, -1.0, float("nan")])
def test_a_conversion_refuses_a_frequency_it_cannot_evaluate_the_index_at(omega):
    with pytest.raises(ModeError, match="omega must be positive"):
        MaterialKerr(n2=1e-20).susceptibility(IndexMaterial(1.45), omega)


def test_a_conversion_refuses_a_material_with_no_linear_index_there():
    """A law that is not a positive real index converts no coefficient."""

    class Opaque:
        """A law whose relative permittivity is negative: no real linear index."""

        def permittivity(self, omega):
            """Return a negative relative permittivity.

            Parameters
            ----------
            omega : array_like
                Frequencies in rad/s.

            Returns
            -------
            numpy.ndarray
                A negative relative permittivity of the input shape.
            """
            return np.full(np.shape(np.asarray(omega, dtype=float)), -2.0 + 0j)

        @property
        def omega_domain(self):
            """Return an unbounded domain.

            Returns
            -------
            tuple of float
                Lower and upper angular frequency in rad/s.
            """
            return 0.0, float("inf")

        @property
        def identity(self):
            """Return a minimal identity.

            Returns
            -------
            dict
                The model name.
            """
            return {"model": "opaque"}

    with pytest.raises(ModeError, match="not a linear index"):
        MaterialKerr(n2=1e-20).susceptibility(Opaque(), OMEGA)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"omega": np.linspace(1e15, 2e15, 3)}, "at least four increasing"),
        ({"omega": np.array([2e15, 1e15, 3e15, 4e15])}, "at least four increasing"),
        ({"omega": np.array([0.0, 1e15, 2e15, 3e15])}, "at least four increasing"),
        ({"z": float("nan")}, "snapshot position must be finite"),
        ({"profile_error": -1.0}, "profile error must be non-negative"),
        ({"profile_error": float("inf")}, "profile error must be non-negative"),
    ],
)
def test_a_sampling_refuses_what_a_reference_cannot_be_rebuilt_from(change, message):
    nodes = ModalNodeRule(np.zeros((1, 2)), np.ones(1), np.zeros(1, dtype=np.int32), {})
    arguments = {
        "omega": np.linspace(1e15, 2e15, 4),
        "z": 0.0,
        "nodes": nodes,
        "degenerate_blocks": (),
        "profile_error": 1e-9,
    }
    with pytest.raises(ModeError, match=message):
        ModalSampling(**{**arguments, **change})


def test_provider_errors(sample):
    provider = make_provider(radius=250e-9)
    nodes = provider.node_rule(OMEGA, n_r=24)
    with pytest.raises(ModeError, match="domain"):
        provider.solve(OMEGA, -1.0, nodes=nodes)
    with pytest.raises(ModeError, match="ownership"):
        provider.solve(
            OMEGA, 0.0, nodes=replace(nodes, region_index=1 - nodes.region_index)
        )
    # A region whose law is not the channels' own index is refused at
    # construction, at the profile wavelength, and not only when a solve is asked
    # for: the region's law is what a declared n2 is converted against.
    wrong = replace(provider.regions[0], material=IndexMaterial(1.6))
    with pytest.raises(ModeError, match="must be the same material"):
        replace(provider, regions=(wrong, provider.regions[1]))
    with pytest.raises(ModeError, match="does not cover the channels"):
        replace(
            provider,
            regions=(
                replace(
                    provider.regions[0],
                    material=_TestLaw(domain=(1.2 * OMEGA, 1.4 * OMEGA)),
                ),
                provider.regions[1],
            ),
        )
    with pytest.raises(ModeError, match="unbounded"):
        replace(provider, regions=())


@dataclass(frozen=True)
class _TestLaw:
    """A real lossless law reading 1.45 at `OMEGA`, with a declared drift and domain."""

    drift: float = 0.0
    domain: tuple[float, float] = (0.0, float("inf"))

    def permittivity(self, omega):
        """Return the relative permittivity, real and positive."""
        values = np.asarray(omega, dtype=float)
        return np.asarray((1.45 * (values / OMEGA) ** self.drift) ** 2, dtype=complex)

    @property
    def omega_domain(self):
        """Return the declared validity interval in rad/s."""
        return self.domain

    @property
    def identity(self):
        """Return the law's structural identity."""
        return {"model": "test-law", "drift": self.drift}


@pytest.mark.parametrize(
    "material,omega",
    [
        # Agrees exactly at the profile wavelength and drifts away from it: the
        # construction check compares one frequency and the eigenproblem is solved
        # at every frequency a run needs.
        (_TestLaw(drift=1e-3), 0.95 * OMEGA),
        # Agrees in its real part at the profile wavelength and is not lossless:
        # the construction check compares the real index and the step-index
        # eigenproblem has no imaginary one. A complex silica of the caller's: an
        # optmat index is held transparent on this route since SEP 3's Q3.
        (IndexMaterial(absorbing_silica()), OMEGA),
    ],
)
def test_a_region_law_the_construction_check_admits_is_checked_again_at_the_solve(
    material, omega
):
    """Two mismatches a single comparison at the profile wavelength cannot see."""
    core = float(np.real(material.permittivity(np.array([OMEGA]))[0])) ** 0.5
    mode = StepIndexMode(
        250e-9, core=core, cladding=1.0, profile_wavelength=2 * np.pi * C / OMEGA
    )
    regions = (
        MaterialRegion("core", 0.0, mode.core_radius, material),
        MaterialRegion("cladding", mode.core_radius, None, IndexMaterial(1.0)),
    )
    provider = guide_module.StepIndexProvider((mode,), regions)
    nodes = provider.node_rule(OMEGA, n_r=24)
    with pytest.raises(ModeError, match="material differs"):
        provider.solve(omega, 0.0, nodes=nodes)


def test_production_only_record_shape_guards(sample):
    # The session hook rejects these shapes before construction. Drive the
    # production post-init validator directly, as the existing mode tests do.
    for electric, message in [
        (np.zeros((2, 2, 1), complex), "three Cartesian"),
        (np.zeros((0, 3, 1), complex), "nodes and modes"),
    ]:
        changed = replace(sample)
        object.__setattr__(changed, "electric", electric)
        with pytest.raises(ModeError, match=message):
            changed.__post_init__()


def test_glass_material_domain_and_absorbing_field_refusal():
    material = IndexMaterial(malitson_silica())
    assert np.real(material.permittivity([OMEGA]))[0] > 1
    assert "SellmeierGlass" in str(material.identity)
    with pytest.raises(ModeError, match="domain"):
        material.permittivity([1.0])
    # A complex silica of the caller's: an optmat index is held transparent (SEP 3,
    # Q3).
    mode = replace(make_provider().modes[0], core=absorbing_silica())
    with pytest.raises(ModeError, match="absorbing"):
        mode.electromagnetic_field(np.zeros((1, 2)), OMEGA)


def test_guide_delegation_and_region_geometry():
    p = make_provider(radius=250e-9)
    guide = ModalGuide(p, ModeCollection(p.modes), p.regions)
    assert guide.omega_domain == p.omega_domain
    assert guide.z_domain == p.z_domain
    assert guide.z_breakpoints == p.z_breakpoints
    assert guide.identity == p.identity
    nodes = p.node_rule(OMEGA, n_r=12)
    assert guide.solve(OMEGA, 0.0, nodes=nodes).physical_id
    wrong = replace(p.regions[0], outer_radius=300e-9)
    with pytest.raises(ModeError, match="region radii"):
        replace(p, regions=(wrong, p.regions[1])).solve(OMEGA, 0.0, nodes=nodes)
    # Identical cross-section limits do not make different material laws one guide.
    with pytest.raises(ModeError, match=r"physical guide|cross-section"):
        replace(p, modes=(p.modes[0], replace(p.modes[0], core=1.46, parity="odd")))


def test_a_guide_refuses_regions_and_channels_that_are_not_its_providers():
    p = make_provider(radius=250e-9)
    collection = ModeCollection(p.modes)
    assert ModalGuide(p, collection, p.regions).regions == p.regions
    # A guide whose regions are not the provider's own describes one geometry and
    # solves another: the provider validates its regions, this pairing does not.
    other = (replace(p.regions[0], name="renamed"), p.regions[1])
    with pytest.raises(ModeError, match="field provider's own material regions"):
        ModalGuide(p, collection, other)
    # Two TE/TM channels against a provider that has one HE11 built silently.
    # A wider core so that both sectors are above cutoff at 835 nm.
    wrong = ModeCollection(
        [
            StepIndexMode(
                1.5e-6,
                core=1.45,
                cladding=1.0,
                profile_wavelength=835e-9,
                kind=kind,
                n=0,
                m=1,
            )
            for kind in ("TE", "TM")
        ]
    )
    with pytest.raises(ModeError, match="order or count"):
        ModalGuide(p, wrong, p.regions)

    # A provider that names no channels cannot have its order checked at all.
    class Nameless:
        @property
        def regions(self):
            return p.regions

        @property
        def identity(self):
            return {"spectral_convention": "internal"}

        @property
        def z_breakpoints(self):
            return p.z_breakpoints

        @property
        def z_domain(self):
            return p.z_domain

        @property
        def omega_domain(self):
            return p.omega_domain

        def node_rule(self, omega, z=0.0, *, n_r=None, n_theta=None):
            return p.node_rule(omega, z, n_r=n_r, n_theta=n_theta)

        def solve(self, omega, z, *, nodes):
            return p.solve(omega, z, nodes=nodes)

    with pytest.raises(ModeError, match="names its channels"):
        ModalGuide(Nameless(), collection, p.regions)


def test_numerical_certificates_fail_closed(monkeypatch):
    p = make_provider(radius=250e-9)
    nodes = p.node_rule(OMEGA, n_r=12)
    monkeypatch.setattr(guide_module, "_power", lambda *args: (1.0, 2.0))
    with pytest.raises(ModeError, match="tail"):
        p.node_rule(OMEGA)
    with pytest.raises(ModeError, match="norm"):
        p.solve(OMEGA, 0.0, nodes=nodes)
    monkeypatch.setattr(vector_module, "_power", lambda *args: (1.0, 2.0))
    with pytest.raises(ModeError, match="norm"):
        p.modes[0].electromagnetic_field(nodes.xy, OMEGA)
    monkeypatch.setattr(vector_module, "PROFILE_MIN_W", 1e6)
    with pytest.raises(ModeError, match="cutoff"):
        p.modes[0].electromagnetic_field(nodes.xy, OMEGA)


def test_a_real_near_cutoff_channel_is_refused_without_a_patched_floor():
    # TE01 sized so that V = 2.5 at its profile wavelength, where the cladding
    # decay w is 0.358 and the field is bound. Evaluated at 866 nm the same guide
    # has V = 2.4105, barely above the j(0,1) = 2.40483 cutoff, and w falls to
    # 0.0635 -- below the declared PROFILE_MIN_W of 0.1, at which the 1e-8 tail
    # would sit beyond 185 core radii and no transverse rule resolves it.
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
    points = np.array([[0.0, 0.0], [radius / 2, 0.0]])
    assert np.all(
        np.isfinite(mode.electromagnetic_field(points, 2 * np.pi * C / 835e-9)[0])
    )
    with pytest.raises(ModeError, match="too close to cutoff"):
        mode.electromagnetic_field(points, 2 * np.pi * C / 866e-9)


def test_the_vector_field_names_and_checks_its_cartesian_positions():
    mode = make_provider(radius=250e-9).modes[0]
    omega = OMEGA
    # The parameter is `xy`, not the polar `points` of the frozen `field`, and a
    # public boundary owes a ModeError rather than the AttributeError a list used
    # to raise inside the field helper.
    from_list = mode.electromagnetic_field([[0.0, 0.0], [1e-7, 0.0]], omega)
    from_array = mode.electromagnetic_field(np.array([[0.0, 0.0], [1e-7, 0.0]]), omega)
    np.testing.assert_array_equal(from_list[0], from_array[0])
    for bad in (np.zeros(2), np.zeros((2, 3)), [0.0, 1.0]):
        with pytest.raises(ModeError, match="Cartesian"):
            mode.electromagnetic_field(bad, omega)


def test_bad_boundary_trace_is_refused(monkeypatch):
    p = make_provider(radius=250e-9)
    nodes = p.node_rule(OMEGA, n_r=12)
    original = guide_module._fields

    def broken(*args, **kwargs):
        e, h = original(*args, **kwargs)
        return (2 * e if kwargs.get("exterior") else e), h

    monkeypatch.setattr(guide_module, "_fields", broken)
    with pytest.raises(ModeError, match="boundary residual"):
        p.solve(OMEGA, 0.0, nodes=nodes)


def test_equal_truncation_does_not_hide_different_guides(monkeypatch):
    p = make_provider(radius=250e-9)
    original_limits = p.modes[0].dimlimits()
    monkeypatch.setattr(
        type(p.modes[0]), "dimlimits", lambda self, z=0.0: original_limits
    )
    with pytest.raises(ModeError, match="physical guide"):
        replace(p, modes=(p.modes[0], replace(p.modes[0], core=1.46, parity="odd")))


def test_scalar_numeric_tower_preserves_frequency_dtype():
    p = make_provider(radius=250e-9)
    nodes = p.node_rule(OMEGA, n_r=12)
    integer = p.solve(int(OMEGA), 0, nodes=nodes)
    floating = p.solve(float(int(OMEGA)), 0.0, nodes=nodes)
    # Scalar ints are accepted as floats by the public numeric-tower contract.
    np.testing.assert_array_equal(integer.electric, floating.electric)
    np.testing.assert_array_equal(integer.generator, floating.generator)


GRIN_OMEGA = 2 * np.pi * C / 1.03e-6


def grin_provider(count=2, elements=32):
    """Return the reference parabolic TE sector."""
    return GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.45 * np.sqrt(0.99)),
        radius=25e-6,
        family="TE",
        count=count,
        elements=elements,
        extent=2.0,
    )


def test_a_guide_is_built_from_a_constructed_graded_index_provider():
    provider = grin_provider()
    omega = np.linspace(0.98 * GRIN_OMEGA, 1.02 * GRIN_OMEGA, 9)
    guide = ModalGuide.from_provider(provider, omega)
    # The guide's channels are the provider's own, in its order and count: nothing
    # here invents a resolution or a second root solve.
    assert [label.display_label for label in guide.mode_collection.channel_labels] == [
        "TE0,1",
        "TE0,2",
    ]
    assert guide.regions == tuple(provider.regions)
    modes = guide.mode_collection.modes
    nodes = provider.node_rule(float(np.median(omega)), 0.0)
    direct = provider.solve(float(omega[2]), 0.0, nodes=nodes)
    for channel, mode in enumerate(modes):
        # 64 epsilon on a propagation constant of about 1e7 1/m: the constructor's
        # own check is the declared 1e-6, and this asserts the far tighter equality
        # a knot actually reaches.
        assert_allclose(
            np.real(mode.neff(omega[2:3])) * float(omega[2]) / C,
            float(np.real(1j * direct.generator[channel, channel])),
            rtol=64 * np.finfo(float).eps,
            atol=0,
        )
        assert mode.z_constant and not mode.scale_invariant
        # The whole protocol, not only what the guide itself calls: n_eff - 1 is
        # the index less one and the derivative is the interpolant's.
        assert_allclose(
            mode.neff_minus_one(omega[2:3]),
            mode.neff(omega[2:3]) - 1.0,
            rtol=0,
            atol=16 * np.finfo(float).eps,
        )
        assert mode.dneff_domega(omega[2:3]).shape == (1,)
        assert mode.azimuthal_order == 0
        assert mode.aeff() > 0
        assert mode.principal_component() is None
        assert mode.channel_label == guide.mode_collection.channel_labels[channel]
    # One cross-section for the set: the rule's declared truncation radius, the
    # provider's extent times its core radius.
    assert modes[0].dimlimits() == modes[1].dimlimits()
    assert modes[0].dimlimits()[2][0] == 2.0 * 25e-6
    with pytest.raises(ModeError, match="provider channel"):
        modes[0].field(np.zeros((3, 2)))
    # A rule that declares no truncation radius is taken at its furthest node.
    bare = ModalGuide.from_provider(
        provider, omega, nodes=replace(nodes, identity={})
    ).mode_collection.modes[0]
    assert bare.dimlimits()[2][0] == pytest.approx(2.0 * 25e-6, rel=2e-3)
    assert bare.dimlimits()[2][0] < 2.0 * 25e-6


def test_the_surrogate_area_of_a_guide_channel_comes_from_the_median_knot():
    provider = grin_provider()
    omega = np.linspace(0.98 * GRIN_OMEGA, 1.02 * GRIN_OMEGA, 9)
    middle = len(omega) // 2
    nodes = provider.node_rule(float(np.median(omega)), 0.0)
    guide = ModalGuide.from_provider(provider, omega, nodes=nodes)
    median = provider.solve(float(omega[middle]), 0.0, nodes=nodes)
    lowest = provider.solve(float(omega[0]), 0.0, nodes=nodes)
    for channel, mode in enumerate(guide.mode_collection.modes):
        # The median knot, the frequency the default node rule is built at, and
        # not the lowest one. 64 epsilon, not equality: the table's sample carries
        # the tracking's unit phase, which |E|^2 is invariant under only to
        # rounding.
        assert_allclose(
            mode.aeff(),
            channel_area(median, nodes, channel),
            rtol=64 * np.finfo(float).eps,
            atol=0,
        )
        # Which knot it is matters: measured 2.03 % between the lowest and the
        # median over this +-2 % band, 25 % over +-20 % and 66 % over +-40 %.
        assert 1.015 < channel_area(lowest, nodes, channel) / mode.aeff() < 1.025
        table = cast("ProviderChannelMode", mode).table
        identity = dict(cast("_SampledIndexTable", table).source_identity)
        assert identity["channel_area_omega"] == float(omega[middle])


def test_a_guide_refuses_a_table_that_does_not_reproduce_its_provider():
    provider = grin_provider()
    # Over a +-40 % band four knots interpolate beta to 2.08e-7 relative at the middle
    # gap but 3.10e-6 at an end gap (measured by an interpolation study): the default
    # budget refuses the four-knot table at an end, names the gap, and admits nine
    # knots, whose worst gap measures 1.48e-7. A declared 1e-12 is a budget no
    # interpolation meets.
    wide = np.linspace(0.6 * GRIN_OMEGA, 1.4 * GRIN_OMEGA, 4)
    with pytest.raises(ModeError, match="held-out frequency"):
        ModalGuide.from_provider(provider, wide, beta_rtol=1e-12)
    with pytest.raises(ModeError, match=r"(first|last) held-out frequency"):
        ModalGuide.from_provider(provider, wide)
    assert ModalGuide.from_provider(provider, wide, beta_rtol=1e-5) is not None
    nine = np.linspace(0.6 * GRIN_OMEGA, 1.4 * GRIN_OMEGA, 9)
    assert ModalGuide.from_provider(provider, nine) is not None
    with pytest.raises(ModeError, match="beta_rtol must be positive"):
        ModalGuide.from_provider(provider, wide, beta_rtol=0.0)


def test_a_moving_guide_is_sampled_from_its_entrance_and_a_snapshot_has_no_path():
    """A moving guide is sampled; what is refused is what means nothing.

    A guide whose geometry moves is sampled along its whole path from the provider's
    own entrance, so a snapshot position is refused by name rather than silently
    ignored; and a guide whose geometry does not move has no path to divide or to
    sample, so the path's two declarations are refused beside it.
    """
    omega = np.linspace(0.98 * GRIN_OMEGA, 1.02 * GRIN_OMEGA, 4)
    smooth = replace(grin_provider(), radius=Linear(25e-6, 26e-6, 1.0))
    with pytest.raises(ModeError, match="leave z at its default"):
        ModalGuide.from_provider(smooth, omega, z=0.5)
    with pytest.raises(ModeError, match="no path to divide"):
        ModalGuide.from_provider(grin_provider(), omega, sections=2)
    with pytest.raises(ModeError, match="no path to divide"):
        ModalGuide.from_provider(grin_provider(), omega, path_rtol=1e-6)
    # The same provider with a constant radius is a snapshot, with no path.
    guide = ModalGuide.from_provider(grin_provider(), omega)
    assert guide.sampling is not None and guide.sampling.path is None


def test_a_guide_refuses_a_provider_that_names_no_channels():
    p = make_provider(radius=250e-9)

    class Nameless:
        regions = p.regions
        z_breakpoints = p.z_breakpoints
        z_domain = p.z_domain
        omega_domain = p.omega_domain

        @property
        def identity(self):
            return {"spectral_convention": "internal"}

        def node_rule(self, omega, z=0.0, *, n_r=None, n_theta=None):
            return p.node_rule(omega, z, n_r=n_r, n_theta=n_theta)

        def solve(self, omega, z, *, nodes):
            return p.solve(omega, z, nodes=nodes)

    with pytest.raises(ModeError, match="names its channels"):
        ModalGuide.from_provider(Nameless(), np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 4))


def test_a_guide_refuses_channels_that_are_not_labels():
    provider = grin_provider(count=1)
    omega = np.linspace(0.98 * GRIN_OMEGA, 1.02 * GRIN_OMEGA, 4)

    class Mislabelled:
        regions = provider.regions
        z_breakpoints = provider.z_breakpoints
        z_domain = provider.z_domain
        omega_domain = provider.omega_domain

        @property
        def identity(self):
            return {**dict(provider.identity), "channels": [42]}

        def node_rule(self, frequency, z=0.0, *, n_r=None, n_theta=None):
            return provider.node_rule(frequency, z, n_r=n_r, n_theta=n_theta)

        def solve(self, frequency, z, *, nodes):
            return provider.solve(frequency, z, nodes=nodes)

    with pytest.raises(ModeError, match="must be a label mapping"):
        ModalGuide.from_provider(cast("ModalProvider", Mislabelled()), omega)


def test_a_provider_channel_mode_refuses_an_impossible_domain_or_area():
    table = grin_provider(count=1)
    omega = np.linspace(0.98 * GRIN_OMEGA, 1.02 * GRIN_OMEGA, 4)
    mode = cast(
        "ProviderChannelMode", ModalGuide.from_provider(table, omega).mode_collection[0]
    )
    for change in ({"outer_radius": 0.0}, {"area": float("inf")}):
        with pytest.raises(ModeError, match="positive and finite"):
            replace(mode, **change)


def test_a_guide_forwards_its_declared_degenerate_blocks_to_the_tracking():
    """The two parities of HE11 are matched inside their block, not by order.

    Tracking rotates freely inside a declared block and one-to-one outside it, and
    the generator is rotated with the fields. Without the keyword a guide had no
    way to say which of its channels are degenerate, so an exactly degenerate pair
    was matched by an ordering its eigenvalues cannot resolve.
    """
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 5)
    plain = ModalGuide.from_provider(provider, omega)
    blocked = ModalGuide.from_provider(provider, omega, degenerate_blocks=((0, 1),))
    labels = [label.display_label for label in blocked.mode_collection.channel_labels]
    assert labels == ["HE1,1:even", "HE1,1:odd"]
    # Both guides build: the keyword changes how the knots are aligned, not which
    # channels exist, and the propagation constants of a degenerate pair are equal
    # either way -- which is exactly why the ordering cannot resolve them.
    for one, other in zip(
        plain.mode_collection.modes, blocked.mode_collection.modes, strict=True
    ):
        assert_allclose(
            one.neff(omega, 0.0), other.neff(omega, 0.0), rtol=1e-12, atol=0
        )


def test_a_guide_refuses_a_degenerate_block_that_is_not_a_channel_set():
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 5)
    with pytest.raises(ModeError, match="disjoint valid channel sets"):
        ModalGuide.from_provider(provider, omega, degenerate_blocks=((0, 5),))


# ------------------------------------------------- the declared change of basis


PAIR_LABELS = (
    ChannelLabel("HE", 1, 1, "even", None, "HE1,1:even"),
    ChannelLabel("HE", 1, 1, "odd", None, "HE1,1:odd"),
)


def test_the_circular_pair_is_the_internal_convention_s_handedness():
    """`e_left = (e_even - i e_odd)/sqrt(2)`, derived from the internal plane wave.

    The internal convention's plane wave is `exp(i(omega t - beta z))`, so a channel
    with amplitude one and nodal rows `(e_even, e_odd)` has the physical transverse
    field `Re[(e_even + a e_odd) exp(i omega t)]`. With `a = -i` that is
    `(cos, sin)`, which turns from x towards y: positive helicity, left circular,
    and the conjugate `(1, i)/sqrt(2)` in the optics convention, which
    `solprop.polarisation` records as positive Stokes V. The test measures the
    rotation rather than restating the matrix.
    """
    basis = circular_pair(PAIR_LABELS, (0, 1))
    left = basis.transform[:, 0]
    times = np.linspace(0.0, 2.0 * np.pi, 9)
    # The physical transverse vector of the left channel, in (x, y), over a cycle.
    field = np.real(left[:, None] * np.exp(1j * times)[None, :]) * np.sqrt(2.0)
    assert_allclose(field[0], np.cos(times), rtol=0, atol=8 * np.finfo(float).eps)
    assert_allclose(field[1], np.sin(times), rtol=0, atol=8 * np.finfo(float).eps)
    # And the optics-convention Jones vector of the same channel has positive V.
    assert float(stokes(np.conj(left))[3]) > 0.0
    assert basis.labels[0].handedness == "left"
    assert basis.labels[1].handedness == "right"
    assert all(label.parity is None for label in basis.labels)


def test_the_circular_pair_is_unitary_and_names_its_block():
    basis = circular_pair(PAIR_LABELS, (0, 1))
    assert basis.gauge_id == "circular(0,1)"
    assert basis.blocks == ((0, 1),)
    assert basis.unitary_defect < guide_module.BASIS_UNITARY_ATOL
    assert not basis.transform.flags.writeable
    identity = dict(basis.identity)
    assert identity["gauge_id"] == "circular(0,1)"
    assert identity["blocks"] == [[0, 1]]
    # The matrix is in the identity by value: two different unitary transforms of
    # one block are two different bases and the block list cannot tell them apart.
    root = float(1.0 / np.sqrt(2.0))
    assert cast("list", identity["transform_imag"])[1] == [-root, root]
    assert [one["handedness"] for one in cast("list", identity["labels"])] == [
        "left",
        "right",
    ]


def test_the_circular_pair_keeps_every_other_channel_where_it_was():
    labels = (*PAIR_LABELS, ChannelLabel("TE", 0, 1, None, None, "TE0,1"))
    basis = circular_pair(labels, (0, 1))
    assert basis.labels[2] is labels[2]
    assert_allclose(basis.transform[2], [0, 0, 1], rtol=0, atol=0)
    assert_allclose(basis.transform[:, 2], [0, 0, 1], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ((0,), "exactly two distinct"),
        ((0, 0), "exactly two distinct"),
        ((0, 1, 2), "exactly two distinct"),
        ((0, 7), "outside the 2 channel"),
        ((1, 0), "even and the odd parity"),
    ],
)
def test_a_circular_pair_refuses_a_block_that_is_not_one(block, message):
    with pytest.raises(ModeError, match=message):
        circular_pair(PAIR_LABELS, block)


def test_a_circular_pair_refuses_two_parities_of_different_families():
    labels = (PAIR_LABELS[0], ChannelLabel("EH", 2, 1, "odd", None, "EH2,1:odd"))
    with pytest.raises(ModeError, match="even and the odd parity"):
        circular_pair(labels, (0, 1))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"transform": np.ones((2, 3), complex)}, "square matrix"),
        ({"transform": np.full((2, 2), np.nan, complex)}, "square matrix"),
        ({"gauge_id": ""}, "non-empty gauge_id"),
        ({"labels": PAIR_LABELS[:1]}, "one label per channel"),
        ({"blocks": ((0,),)}, "disjoint blocks"),
        ({"blocks": ((0, 1), (1, 0))}, "disjoint blocks"),
        ({"blocks": ((0, 5),)}, "disjoint blocks"),
        ({"blocks": ()}, "identity outside its declared blocks"),
    ],
)
def test_a_channel_basis_refuses_what_it_cannot_describe(changes, message):
    basis = circular_pair(PAIR_LABELS, (0, 1))
    fields = {
        "transform": basis.transform,
        "blocks": basis.blocks,
        "labels": basis.labels,
        "gauge_id": basis.gauge_id,
    } | changes
    with pytest.raises(ModeError, match=message):
        ChannelBasis(**fields)


def test_a_nonunitary_declaration_is_refused_by_name():
    """A skewed basis is not a change of gauge: its metric is genuinely off I."""
    skewed = np.array([[1.0, 0.3], [0.0, 1.0]], dtype=complex)
    with pytest.raises(ModeError, match="must be unitary"):
        ChannelBasis(skewed, ((0, 1),), PAIR_LABELS, "skew")


def test_a_channel_basis_that_mixes_an_undeclared_channel_is_refused():
    labels = (*PAIR_LABELS, ChannelLabel("TE", 0, 1, None, None, "TE0,1"))
    matrix = np.eye(3, dtype=complex)
    matrix[np.ix_([0, 2], [0, 2])] = np.array([[0, 1], [1, 0]], dtype=complex)
    with pytest.raises(ModeError, match="identity outside its declared blocks"):
        ChannelBasis(matrix, ((0, 1),), labels, "swap")


def test_provider_channels_reads_a_provider_s_own_labels():
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    labels = provider_channels(provider)
    assert [one.display_label for one in labels] == ["HE1,1:even", "HE1,1:odd"]


def relabelled_provider(provider, channels):
    """Return the provider with `channels` swapped into its published identity."""

    class Relabelled:
        regions = provider.regions
        z_breakpoints = provider.z_breakpoints
        z_domain = provider.z_domain
        omega_domain = provider.omega_domain

        @property
        def identity(self):
            declared = dict(provider.identity)
            if channels is None:
                declared.pop("channels")
            else:
                declared["channels"] = channels
            return declared

        def node_rule(self, frequency, z=0.0, *, n_r=None, n_theta=None):
            return provider.node_rule(frequency, z, n_r=n_r, n_theta=n_theta)

        def solve(self, frequency, z, *, nodes):
            return provider.solve(frequency, z, nodes=nodes)

    return cast("ModalProvider", Relabelled())


@pytest.mark.parametrize(
    ("channels", "message"),
    [(None, "names no channels"), ([], "names no channels"), ([42], "label mapping")],
)
def test_provider_channels_refuses_an_identity_it_cannot_read(channels, message):
    provider = make_provider()
    with pytest.raises(ModeError, match=message):
        provider_channels(relabelled_provider(provider, channels))


def test_a_guide_in_a_declared_basis_carries_the_new_labels_and_the_declaration():
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 5)
    basis = circular_pair(provider_channels(provider), (0, 1))
    guide = ModalGuide.from_provider(
        provider, omega, degenerate_blocks=((0, 1),), basis=basis
    )
    sampling = guide.sampling
    assert sampling is not None and sampling.basis is basis
    labels = [one.display_label for one in guide.mode_collection.channel_labels]
    assert labels == ["HE1,1:left", "HE1,1:right"]
    # The declaration is in the identity only where one is declared, so a guide in
    # the provider's own coordinates keeps the identity it has always had.
    assert "basis" in dict(sampling.identity)
    plain = ModalGuide.from_provider(provider, omega, degenerate_blocks=((0, 1),))
    assert plain.sampling is not None
    assert "basis" not in dict(plain.sampling.identity)
    # The two circular partners of an exactly degenerate pair share its beta --
    # exactly, because `apply_channel_basis` carries the generator over (a copy,
    # a structural contract) rather than evaluating `U.H G U`, and the index table
    # is built from that diagonal.
    for one, other in zip(
        plain.mode_collection.modes, guide.mode_collection.modes, strict=True
    ):
        assert np.array_equal(one.neff(omega, 0.0), other.neff(omega, 0.0))


def test_a_guide_refuses_a_basis_over_an_undeclared_degenerate_block():
    provider = make_provider(descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")))
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 5)
    basis = circular_pair(provider_channels(provider), (0, 1))
    with pytest.raises(ModeError, match="do not name"):
        ModalGuide.from_provider(provider, omega, basis=basis)


def test_a_guide_refuses_a_basis_that_renames_a_channel_it_does_not_mix():
    """Defence in depth: the declaration owns its blocks' labels and no others.

    ``from_provider`` adopts the declaration's labels, and ``__post_init__``
    compares the guide against those same labels, so a hand-built record that
    relabelled a channel outside its blocks would rename it with nothing left to
    notice. The channels the transform is the identity on keep the provider's own.
    """
    descriptors = (("HE", 1, 1, "even"), ("HE", 1, 1, "odd"), ("TE", 0, 1, "even"))
    provider = make_provider(descriptors=descriptors)
    omega = np.linspace(0.99 * OMEGA, 1.01 * OMEGA, 5)
    basis = circular_pair(provider_channels(provider), (0, 1))
    relabelled = (
        *basis.labels[:2],
        ChannelLabel("TE", 0, 1, None, None, "TE0,1:renamed"),
    )
    hand_built = ChannelBasis(basis.transform, basis.blocks, relabelled, basis.gauge_id)
    with pytest.raises(ModeError, match="renames channel 2, which it does not mix"):
        ModalGuide.from_provider(
            provider, omega, degenerate_blocks=((0, 1),), basis=hand_built
        )
    # The declaration `circular_pair` builds carries the provider's own label
    # there, so the same guide is admitted and the TE channel keeps its name.
    guide = ModalGuide.from_provider(
        provider, omega, degenerate_blocks=((0, 1),), basis=basis
    )
    labels = [one.display_label for one in guide.mode_collection.channel_labels]
    assert labels == ["HE1,1:left", "HE1,1:right", "TE0,1"]

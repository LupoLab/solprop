# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Independent phase, ray geometry and physical-domain tests for compressor pairs."""

import inspect
import math
from dataclasses import replace

import mpmath as mp
import numpy as np
import pytest

from compressor_reference import (
    MALITSON_SILICA,
    grating_wavefront,
    prism_reference,
    reference_phase,
    silica,
)
from solprop.constants import C
from solprop.exceptions import MaterialError, OpticsGeometryError
from solprop.optics import (
    Chain,
    GratingPair,
    PrismGeometry,
    PrismPair,
    TaylorPhase,
    _prism_intersect,
    _prism_refract,
    brewster_angle,
    brewster_apex,
    littrow_angle,
    stored_factor,
)
from solprop.physdata.glasses import GlassIndex
from solprop.processing._examples import example_grid

WAVELENGTH = 800e-9
OMEGA0 = 2 * math.pi * C / WAVELENGTH
SILICA = MALITSON_SILICA


def prism(*, brewster=True, insertion=0.015, separation=0.6, band=None):
    angles = (
        (brewster_apex(SILICA, WAVELENGTH), brewster_angle(SILICA, WAVELENGTH))
        if brewster
        else (math.radians(70), math.radians(55))
    )
    return PrismPair(
        SILICA,
        PrismGeometry(separation, insertion, 0.003, *angles),
        WAVELENGTH,
        band=band,
    )


def grating(**kwargs):
    return replace(
        GratingPair(1.2e6, 0.05, littrow_angle(1.2e6, WAVELENGTH), WAVELENGTH),
        **kwargs,
    )


PHASE_ROUNDING = 1e-12
"""Agreement of two evaluations of one pair's phase, as a fraction of its peak.

Both are the same elementwise arithmetic, so they differ by rounding alone: a few dozen
roundings (64 eps) of terms that the group-delay subtraction cancels down to the
residual, by up to a factor of about seventy over a +-20 % band (the residual is second
order in the exit-angle change of about 1e-2 rad, its terms first order).
"""


def assert_same_phase(actual, expected):
    """Assert two evaluations of a pair's phase agree to :data:`PHASE_ROUNDING`."""
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0,
        atol=PHASE_ROUNDING * float(np.max(np.abs(expected))),
    )


@pytest.mark.parametrize("order,incidence", [(-1, 0.45), (1, -0.55), (-2, 0.8)])
@pytest.mark.parametrize("double_pass", [False, True])
def test_grating_phase_matches_treacy(order, incidence, double_pass):
    pair = grating(
        line_density=600e3,
        order=order,
        incidence_angle=incidence,
        double_pass=double_pass,
    )
    omega = 2 * math.pi * C / np.linspace(600e-9, 1000e-9, 11)
    # 50-digit ray+groove reference, compared at double input coordinates. At most
    # 1e7 raw radians: 5e-8 rad covers the independent reference/frame rounding.
    np.testing.assert_allclose(
        pair.phase(omega), reference_phase(pair, omega), rtol=0, atol=5e-8
    )


def test_grating_gdd_tod_independent():
    pair = grating()
    with mp.workdps(90):
        at = 2 * mp.pi * C / mp.mpf(WAVELENGTH)
        gdd = float(mp.diff(lambda w: grating_wavefront(pair, w), at, 2))
        tod = float(mp.diff(lambda w: grating_wavefront(pair, w), at, 3))
    _derivative_convergence(pair, gdd, order=2)
    _derivative_convergence(pair, tod, order=3)
    density, wavelength = pair.line_density, pair.wavelength
    sine = math.sin(pair.incidence_angle)
    cosine = math.sqrt(1 - (sine + pair.order * wavelength * density) ** 2)
    analytic = (
        -2
        * pair.separation
        * pair.order**2
        * wavelength**3
        * density**2
        / (2 * math.pi * C**2 * cosine**3)
    )
    ratio = (
        -3
        * wavelength
        / (2 * math.pi * C)
        * (1 - pair.order * wavelength * density * sine - sine**2)
        / cosine**2
    )
    # Closed forms differ only in elementary-function rounding (tens of ulps).
    assert gdd == pytest.approx(analytic, rel=2e-14)
    assert tod == pytest.approx(gdd * ratio, rel=2e-14)
    assert abs(tod - gdd * 2 * ratio) > 0.9 * abs(tod)  # Keller (3.42) factor two.
    # D is normal spacing: the ray's slant length is D/cos(theta_m).
    assert pair.separation / cosine > pair.separation
    assert littrow_angle(1200 * 1000, WAVELENGTH) == pair.incidence_angle
    # A pair with no separation has no phase: a product with zero, exactly.
    np.testing.assert_array_equal(
        replace(pair, separation=0.0).phase([OMEGA0 * 0.9, OMEGA0]), 0.0
    )


@pytest.mark.parametrize("brewster", [True, False])
def test_prism_phase_matches_wavefront_ray_trace(brewster):
    pair = prism(brewster=brewster)
    omega = 2 * math.pi * C / np.linspace(650e-9, 1050e-9, 13)
    np.testing.assert_allclose(
        pair.phase(omega), reference_phase(pair, omega), rtol=0, atol=5e-8
    )
    paths = pair.ray_paths(omega)
    with mp.workdps(90):
        _, _, ray, *_ = prism_reference(pair)
        for position, value in enumerate(omega):
            traced = ray(mp.mpf(float(value)))
            expected = [float(x) for x in (*traced["lengths"], *traced["faces"])]
            actual = [
                getattr(paths, name)[position] for name in paths.__dataclass_fields__
            ]
            # Ray intersection errors scale with metre paths / sin(face angle).
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-15)
            assert min(actual) > 0
            assert mp.norm(traced["output"] - traced["incident"]) < mp.mpf("1e-45")


@pytest.mark.parametrize("brewster", [True, False])
def test_prism_gdd_matches_arissian_diels(brewster):
    pair = prism(brewster=brewster)
    with mp.workdps(90):
        at, raw, ray, gdd7, *_ = prism_reference(pair)
        reference = mp.diff(raw, at, 2)
        assert abs(gdd7(at) / reference - 1) < mp.mpf("1e-40")
        if brewster:
            wavelength = 2 * mp.pi * C / at
            index = silica(at)
            n_l = mp.diff(lambda w: silica(2 * mp.pi * C / w), wavelength)
            n_ll = mp.diff(lambda w: silica(2 * mp.pi * C / w), wavelength, 2)
            paths = ray(at)
            eq10 = (
                2
                * wavelength**3
                / (2 * mp.pi * C**2)
                * (
                    paths["glass"] * n_ll
                    - (4 * paths["air"] + paths["glass"] / index**3) * n_l**2
                )
            )
            # Double-precision Brewster angles differ from exact ones by ~1e-16 rad.
            assert float(eq10 / reference) == pytest.approx(1.0, rel=2e-13)
    _derivative_convergence(pair, float(reference), order=2)


def _derivative_convergence(pair, expected, *, order):
    steps = np.array([0.012, 0.006, 0.003])
    estimates = []
    for step in steps:
        values = pair.phase(OMEGA0 * (1 + step * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])))
        if order == 2:
            estimate = (values[3] - 2 * values[2] + values[1]) / (OMEGA0 * step) ** 2
        else:
            estimate = (values[4] - 2 * values[3] + 2 * values[1] - values[0]) / (
                2 * (OMEGA0 * step) ** 3
            )
        estimates.append(estimate)
    errors = np.abs(np.array(estimates) - expected)
    # Centred differences are second order; three levels must approach a factor 4
    # before the floating-point floor. Richardson removes their leading h^2 term.
    assert np.all(errors[:-1] / errors[1:] > 3.5)
    extrapolated = (4 * estimates[-1] - estimates[-2]) / 3
    assert extrapolated == pytest.approx(expected, rel=1e-6 if order == 2 else 1e-5)


@pytest.mark.parametrize("brewster", [True, False])
def test_prism_tod_is_derivative_of_full_gdd(brewster):
    pair = prism(brewster=brewster)
    with mp.workdps(90):
        at, raw, ray, gdd7, angles, _ = prism_reference(pair)
        interior, exterior = angles
        n, n1, n2, n3 = [mp.diff(silica, at, k) for k in range(4)]
        glass, air = ray(at)["glass"], ray(at)["air"]
        glass1 = mp.diff(lambda w: ray(w)["glass"], at)
        e1, e2 = mp.diff(exterior, at), mp.diff(exterior, at, 2)
        r1, r2 = mp.diff(interior, at), mp.diff(interior, at, 2)
        derivative = (
            mp.mpf(2)
            / C
            * (
                glass1 * (2 * n1 + at * n2)
                + glass * (3 * n2 + at * n3)
                - air * e1**2
                - at * air * mp.tan(exterior(at)) * e1**3
                - 2 * at * air * e1 * e2
                - (n * glass + at * n1 * glass + n * at * glass1) * r1**2
                - 2 * n * at * glass * r1 * r2
            )
        )
        reference = mp.diff(raw, at, 3)
        # Full product-rule derivative, including both length derivatives, agrees at
        # 40 digits using 90-digit arithmetic: third differences at optical omega
        # lose about 34 digits. Printed Eq. (11) misses the final factor two.
        assert abs((derivative - reference) / reference) < mp.mpf("1e-40")
        assert abs((mp.diff(gdd7, at) - reference) / reference) < mp.mpf("1e-40")
        assert abs(2 * n * at * glass * r1 * r2 / C) > abs(reference) * mp.mpf("1e-5")
    _derivative_convergence(pair, float(reference), order=3)


def test_prism_insertion_moves_only_second_prism():
    pair = prism()
    moved = replace(pair, geometry=replace(pair.geometry, insertion=0.016))
    along = replace(pair, geometry=replace(pair.geometry, separation=0.61))
    with mp.workdps(90):
        at, _, ray, _, _, apex = prism_reference(pair)
        _, _, moved_ray, _, _, moved_apex = prism_reference(moved)
        _, _, _, _, _, along_apex = prism_reference(along)
        np.testing.assert_allclose(
            [float(x) for x in moved_apex - apex], [0.0, -0.001], rtol=0, atol=3e-18
        )
        # A separation motion follows the fixed reference air ray at angle -20 deg.
        displacement = np.array([float(x) for x in along_apex - apex])
        assert np.linalg.norm(displacement) == pytest.approx(0.01, rel=2e-14)
        assert displacement[0] > 0 and displacement[1] < 0
        assert ray(at)["lengths"][0] == moved_ray(at)["lengths"][0]
    assert pair.geometry.separation == moved.geometry.separation
    assert moved.apex_distance > pair.apex_distance
    before, after = pair.ray_paths([OMEGA0]), moved.ray_paths([OMEGA0])
    # The first prism's path is formed again from inputs the move does not touch: the
    # same intersections, to their rounding (the 2e-15 m of the ray tests below).
    np.testing.assert_allclose(
        before.first_glass, after.first_glass, rtol=0, atol=2e-15
    )
    assert after.second_glass[0] > before.second_glass[0]
    assert after.air_gap[0] < before.air_gap[0]


def test_prism_first_face_distance_changes_feasibility_not_dispersion():
    pair = prism()
    different = replace(
        pair, geometry=replace(pair.geometry, first_face_distance=0.006)
    )
    omega = OMEGA0 * np.array([0.8, 1.0, 1.2])
    # The phase does not depend on the distance at all: the same arithmetic on the
    # same inputs, to a few roundings of its largest term (PHASE_ROUNDING).
    assert_same_phase(different.phase(omega), pair.phase(omega))
    first, second = pair.ray_paths(omega), different.ray_paths(omega)
    assert np.all(second.first_glass > first.first_glass)
    assert np.all(second.second_glass < first.second_glass)
    # Changing a reallocates fixed total glass; ray intersections cost ~1e-15 m.
    np.testing.assert_allclose(
        first.first_glass + first.second_glass,
        second.first_glass + second.second_glass,
        rtol=0,
        atol=2e-15,
    )
    with pytest.raises(OpticsGeometryError, match="non-forward"):
        replace(pair, geometry=replace(pair.geometry, insertion=0.0))


class ConstantGlass(GlassIndex):
    def _index_and_derivative(self, omega):
        return np.full(omega.shape, 1.45 + 0j), np.zeros(omega.shape, dtype=complex)


def test_constant_index_prisms_have_no_residual_phase():
    medium = ConstantGlass(
        name="constant",
        source="analytic test",
        wavelength_range=(200e-9, 3e-6),
        citation="constant index",
    )
    pair = PrismPair(
        medium,
        PrismGeometry(
            0.6,
            0.015,
            0.003,
            brewster_apex(medium, WAVELENGTH),
            brewster_angle(medium, WAVELENGTH),
        ),
        WAVELENGTH,
    )
    # Without dispersion the exit angle never leaves its reference: an exact zero.
    np.testing.assert_array_equal(pair.phase(OMEGA0 * np.array([0.8, 1.0, 1.2])), 0.0)
    assert repr(pair)
    assert str(pair.ray_paths([OMEGA0]))
    assert isinstance(PrismPair("SiO2", pair.geometry, WAVELENGTH).medium, str)


@pytest.mark.parametrize("pair", [grating(), prism()])
def test_pair_phase_frame_is_order_independent(pair):
    omega = OMEGA0 * np.array([1.1, 0.98, 1.0, 0.8, 1.2])
    phases = pair.phase(omega)
    # At the reference the residual is a product with a zero detuning: exactly zero.
    assert phases[2] == 0.0
    # Each frequency's phase is its own: the same elementwise arithmetic in any order
    # or selection, to its rounding (PHASE_ROUNDING).
    assert_same_phase(pair.phase(omega[[3, 0]]), phases[[3, 0]])
    assert_same_phase(pair.phase([omega[1]]), phases[[1]])
    assert_same_phase(pair.phase(omega[::-1]), phases[::-1])
    assert_same_phase(2 * replace(pair, double_pass=False).phase(omega), phases)
    # Symmetric slope divided by detuning is O(h^2); the absolute residual error
    # floor is a few 1e-10 rad for the prism, so use h=1e-5 and a 0.05 fs GD bound.
    step = OMEGA0 * 1e-5
    around = pair.phase([OMEGA0 - step, OMEGA0 + step])
    assert abs((around[1] - around[0]) / (2 * step)) < 5e-17
    response = pair(omega)
    np.testing.assert_allclose(
        np.abs(response), 1.0, rtol=4 * np.finfo(float).eps, atol=0
    )
    assert pair(np.array([])).shape == (0,)
    assert pair([0.0, -1.0]).tolist() == [0j, 0j]
    # An empty Taylor phase is a factor of exactly one: an exact no-op in a chain.
    np.testing.assert_array_equal(
        Chain((pair, TaylorPhase((), WAVELENGTH)))(omega), response
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1 + 1j]),
        np.ones((1, 1)),
        np.array(1.0),
        ["bad"],
    ],
)
@pytest.mark.parametrize("pair", [grating(), prism()])
def test_pair_frequency_domain_errors(pair, bad):
    with pytest.raises(MaterialError):
        pair(bad)
    with pytest.raises(MaterialError):
        pair.phase(bad)


@pytest.mark.parametrize("pair", [grating(), prism()])
def test_fixed_band_selects_without_tracing_outside(pair):
    selected = replace(pair, band=(700e-9, 900e-9))
    # A very long wavelength has no grating order, but it is outside the fixed band.
    omega = np.array([0.0, -1.0, OMEGA0 * 0.01, OMEGA0, OMEGA0 * 20])
    # Zero outside the band (a hard selection) and exp(0) at the reference: exact.
    np.testing.assert_array_equal(selected(omega), [0j, 0j, 0j, 1 + 0j, 0j])
    with pytest.raises(MaterialError, match="in-band"):
        selected.phase(omega)
    if isinstance(selected, PrismPair):
        with pytest.raises(MaterialError, match="in-band"):
            selected.ray_paths(omega)
    grid = example_grid()
    factor = stored_factor(grid, selected)
    np.testing.assert_array_equal(factor[~grid.active_mask], 1.0)
    for bad in [(0.0, 900e-9), (900e-9, 700e-9), (850e-9, 900e-9), (700e-9, math.inf)]:
        with pytest.raises(MaterialError, match="band"):
            replace(pair, band=bad)
    for bad in [0.0, -1.0, math.nan, math.inf]:
        with pytest.raises(MaterialError, match="wavelength"):
            replace(pair, wavelength=bad)


@pytest.mark.parametrize(
    "settings",
    [
        {"line_density": 0.0},
        {"line_density": math.inf},
        {"order": 0},
        {"order": True},
        {"separation": -1.0},
        {"separation": math.nan},
        {"incidence_angle": math.pi / 2},
        {"incidence_angle": math.nan},
    ],
)
def test_grating_parameter_errors(settings):
    with pytest.raises(MaterialError):
        grating(**settings)


def test_grating_order_boundaries_and_endpoint_checks():
    # Zero incidence, first order: lambda*rho=1 is exactly the grazing boundary.
    for ratio in [1.0, 1.0 + 1e-10]:
        with pytest.raises(OpticsGeometryError):
            grating(line_density=ratio / WAVELENGTH, incidence_angle=0.0, order=1)
    pair = grating(
        line_density=(1.0 - 1e-10) / WAVELENGTH, incidence_angle=0.0, order=1
    )
    assert pair.phase([OMEGA0])[0] == 0.0
    with pytest.raises(OpticsGeometryError):
        pair([OMEGA0 * 0.999])
    with pytest.raises(OpticsGeometryError):
        replace(pair, band=(700e-9, 900e-9))
    with pytest.raises(OpticsGeometryError):
        littrow_angle(2 / WAVELENGTH, WAVELENGTH)


@pytest.mark.parametrize(
    "settings",
    [
        {"separation": -1.0},
        {"insertion": -1.0},
        {"first_face_distance": 0.0},
        {"first_face_distance": math.inf},
        {"apex_angle": math.pi},
        {"apex_angle": math.nan},
        {"incidence_angle": 0.0},
        {"incidence_angle": math.pi / 2},
    ],
)
def test_prism_geometry_parameter_errors(settings):
    with pytest.raises(MaterialError):
        replace(prism().geometry, **settings)


def test_prism_geometry_domain_errors():
    pair = prism()
    half = pair.geometry.apex_angle / 2
    minimum = pair.geometry.first_face_distance / (2 * math.cos(half))
    with pytest.raises(OpticsGeometryError):
        replace(pair, geometry=replace(pair.geometry, insertion=minimum - 1e-9))
    valid = replace(pair, geometry=replace(pair.geometry, insertion=minimum + 1e-9))
    assert valid.ray_paths([OMEGA0]).second_glass[0] > 0
    with pytest.raises(OpticsGeometryError):
        replace(
            pair,
            geometry=replace(pair.geometry, separation=pair.geometry.insertion - 1e-9),
        )
    valid = replace(
        pair, geometry=replace(pair.geometry, separation=pair.geometry.insertion + 1e-9)
    )
    assert valid.ray_paths([OMEGA0]).air_gap[0] > 0
    with pytest.raises(OpticsGeometryError, match="reflecting"):
        replace(pair, geometry=replace(pair.geometry, apex_angle=math.radians(100)))


@pytest.mark.parametrize("factory", [grating, prism])
def test_pass_count_requires_a_boolean(factory):
    with pytest.raises(MaterialError, match="boolean"):
        pair = factory()
        # Bypass only the test session's generated constructor type wrapper,
        # so the production boundary validation itself is exercised.
        values = {
            name: getattr(pair, name)
            for name, spec in pair.__dataclass_fields__.items()
            if spec.init
        }
        values["double_pass"] = 1
        inspect.unwrap(type(pair).__init__)(pair, **values)


@pytest.mark.parametrize("value", [0.0, 1.0, math.inf, math.nan])
def test_bad_real_glass_index_is_a_material_error(value):
    class BadGlass(ConstantGlass):
        def _index_and_derivative(self, omega):
            return np.full(omega.shape, complex(value)), np.zeros(omega.shape, complex)

    medium = BadGlass(
        name="bad", source="test", wavelength_range=(200e-9, 3e-6), citation="test"
    )
    with pytest.raises(MaterialError, match="index"):
        brewster_angle(medium, WAVELENGTH)
    with pytest.raises(MaterialError, match="index"):
        replace(prism(), medium=medium)


def test_nonfinite_index_derivative_is_not_a_geometry_failure():
    class BadDerivative(ConstantGlass):
        def _index_and_derivative(self, omega):
            return np.full(omega.shape, 1.45 + 0j), np.full(
                omega.shape, complex(math.nan)
            )

    medium = BadDerivative(
        name="bad", source="test", wavelength_range=(200e-9, 3e-6), citation="test"
    )
    with pytest.raises(MaterialError, match="derivative") as caught:
        replace(prism(), medium=medium)
    assert not isinstance(caught.value, OpticsGeometryError)


def test_forward_snell_and_grazing_intersection_boundaries():

    # A perpendicular outgoing face normal would require propagation backwards.
    with pytest.raises(OpticsGeometryError, match="backwards"):
        _prism_refract(np.array([[-1.0, 0.0]]), np.array([1.0, 0.0]), np.array([1.0]))
    # Parallel ray and face lines never define a forward intersection.
    with pytest.raises(OpticsGeometryError, match="grazes"):
        _prism_intersect(
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 1.0]]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
        )


def test_pair_rejects_unrepresentable_frequency_and_phase():
    with pytest.raises(MaterialError, match="unrepresentable"):
        grating(wavelength=1e-320)
    with pytest.raises(MaterialError, match="unrepresentable"):
        replace(prism(), band=(1e-320, 1e-6))
    with pytest.raises(MaterialError, match="phase is not finite"):
        grating(separation=1e308).phase([OMEGA0 * 1.1])
    with pytest.raises(OpticsGeometryError, match="order"):
        grating()([1e-320])

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupling models of ``couple`` against analytic and independent references.

SEP 31 sections 4.3, 4.4 and 7: the Gaussian overlap formulas, an independent
Gaussian--Bessel quadrature, nested mode sets with domain, quadrature and modal
refinement kept apart, poor and zero capture with nothing restored, rotated
degenerate bases, coherent interference and gauge invariance, a change of capillary
radius by direct overlap and by synthesis then projection, the passive-map
certificate, and the refusals. A propagation is ten steps at most.
"""

import dataclasses
import itertools
import math
from collections.abc import Sequence

import numpy as np
import pytest
from scipy.special import j0, j1, jn_zeros

from solprop import fourier
from solprop.constants import EPS0, C
from solprop.coupling import (
    STEP_INDEX_BEAM_ENVELOPE,
    STEP_INDEX_ENVELOPE,
    BasisChange,
    CouplingModel,
    FieldPlane,
    InputPort,
    Outcouple,
    PlaneBasis,
    PlaneGeometry,
    PlaneMetric,
    PlaneReference,
    ScalarModeBasis,
    basis_identity,
    couple,
)
from solprop.coupling.coupler import _contrast_items, _gain_item, _whitened_extremes
from solprop.coupling.models import cone_angle, radial_rule
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import CouplingError
from solprop.grid import FieldGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode, Mode, StepIndexMode
from solprop.processing import field_plane
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController

RADIUS = 75e-6
PRESSURE = 3.0
GRID = {
    "wavelength": 800e-9,
    "wavelength_limits": (700e-9, 920e-9),
    "trange": 200e-15,
}
MEDIUM = SpatialMedium("Ar", PRESSURE)
SCALAR_BUDGET = 0.05
SCALAR = CouplingModel("scalar_paraxial", model_error=SCALAR_BUDGET)
SAME = CouplingModel("same_basis")


def capillary(radius: float = RADIUS, modes: int | Sequence[Mode] = 4) -> Simulation:
    """Return a short argon capillary simulation on the tests' envelope grid."""
    return Simulation.from_capillary(
        radius,
        1e-4,
        "Ar",
        PRESSURE,
        modes=modes,
        n_save=2,
        noise=None,
        envelope=True,
        energy=1e-6,
        fwhm=30e-15,
        **GRID,
    )


@pytest.fixture(scope="module")
def port():
    """The 75 um capillary's port with four HE1m modes."""
    return capillary().input_port()


def spectral(grid, omega):
    """Return a 30 fs Gaussian spectrum about the carrier, V s/m per unit field."""
    width = 30e-15 / (2 * math.sqrt(math.log(2)))
    return 1e-9 * np.exp(-(((omega - grid.omega0) * width) ** 2) / 2)


def radial_beam(grid, waist):
    """Return an x-polarised Gaussian of 1/e field radius `waist` on a radial grid."""

    def field(r, omega):
        return np.exp(-((r / waist) ** 2))[:, None] * spectral(grid, omega)[None, :]

    return field


def cartesian_beam(
    grid, waist, *, shift=0.0, tilt=0.0, curvature=math.inf, jones=(1, 0)
):
    """Return a displaced, tilted, curved Gaussian on a Cartesian grid.

    The field is ``exp(-((x - d)^2 + y^2)/w^2) exp(i k tilt x) exp(-i k r^2 / 2R)``,
    in the optics convention, with Jones vector ``jones``.
    """

    def field(x, y, omega):
        k = omega / C
        spatial = np.exp(-((x - shift) ** 2 + y**2) / waist**2)[:, None]
        phase = np.exp(1j * np.outer(x, k) * tilt)
        if math.isfinite(curvature):
            phase = phase * np.exp(-1j * np.outer(x**2 + y**2, k) / (2 * curvature))
        values = spatial * phase * spectral(grid, omega)[None, :]
        return np.stack([jones[0] * values, jones[1] * values], axis=1)

    return field


def shares(result, grid):
    """Return each launched channel's energy over the incident energy, all bins."""
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    launched = np.sum(weights * np.abs(result.launch.samples[:, 0, :]) ** 2, axis=1)
    return launched / result.report.incident_energy


# ------------------------------------------------------------- analytic overlaps


def gaussian_efficiency(w1, w2, shift, tilt_kappa, curvature_k=0.0):
    """Return the section 4.4 capture of two scalar Gaussians, in closed form."""
    a1 = 1 / w1**2
    a2 = 1 / w2**2 + 1j * curvature_k
    aligned = 4 * a1 * a2.real / abs(np.conj(a1) + a2) ** 2
    if curvature_k == 0.0:
        aligned = (2 * w1 * w2 / (w1**2 + w2**2)) ** 2
    return (
        aligned
        * math.exp(-2 * shift**2 / (w1**2 + w2**2))
        * math.exp(-(tilt_kappa**2) * w1**2 * w2**2 / (2 * (w1**2 + w2**2)))
    )


@pytest.mark.parametrize(
    ("w2", "shift", "tilt", "curvature"),
    [
        (30e-6, 0.0, 0.0, math.inf),
        (45e-6, 0.0, 0.0, math.inf),
        (30e-6, 12e-6, 0.0, math.inf),
        (30e-6, 0.0, 4e-3, math.inf),
        (30e-6, 0.0, 0.0, 0.02),
    ],
)
def test_the_gaussian_overlap_formulas_hold_against_direct_integration(
    port, w2, shift, tilt, curvature
):
    """Waist, displacement, tilt and curvature: formula, direct sum, plane overlap.

    The closed forms of section 4.4 are checked against a direct two-dimensional
    sum over a fine grid, and the same overlap of two sampled Cartesian planes
    (``FieldPlane.from_analytic``) in their spectral quadrature checks the sampling,
    the sign of the tilt phase and the amplitude.
    """
    grid = port.grid
    w1 = 30e-6
    k_bin = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    k = float(grid.omega[k_bin]) / C
    curvature_k = 0.0 if not math.isfinite(curvature) else k / (2 * curvature)
    formula = gaussian_efficiency(w1, w2, shift, k * tilt, curvature_k)
    axis = np.linspace(-300e-6, 300e-6, 1201)
    x, y = np.meshgrid(axis, axis)
    u1 = np.exp(-(x**2 + y**2) / w1**2)
    u2 = np.exp(-((x - shift) ** 2 + y**2) / w2**2) * np.exp(1j * k * tilt * x)
    u2 = u2 * np.exp(-1j * curvature_k * (x**2 + y**2))
    direct = abs(np.sum(u1.conj() * u2)) ** 2 / (
        np.sum(abs(u1) ** 2) * np.sum(abs(u2) ** 2)
    )
    # A uniform sum of rapidly decaying Gaussians is spectrally accurate.
    assert direct == pytest.approx(formula, rel=1e-10, abs=1e-14)
    space = FreeGrid(300e-6, 128, 300e-6, 128)
    index = np.where(grid.omega > 0, 1.0, 0.0)
    first = FieldPlane.from_analytic(
        grid, PlaneGeometry(0.0), cartesian_beam(grid, w1), space=space, index=index
    )
    second = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        cartesian_beam(grid, w2, shift=shift, tilt=tilt, curvature=curvature),
        space=space,
        index=index,
    )
    weights = space.spectral_weights
    a = fourier.spectrum_to_internal(first.samples, kind=grid.kind)[:, 0, k_bin]
    b = fourier.spectrum_to_internal(second.samples, kind=grid.kind)[:, 0, k_bin]
    planar = abs(np.sum(weights * a.conj() * b)) ** 2 / (
        np.sum(weights * abs(a) ** 2) * np.sum(weights * abs(b) ** 2)
    )
    # Both beams decay far inside the window and are resolved by its spacing.
    assert planar == pytest.approx(formula, rel=1e-9, abs=1e-12)


def bessel_shares(waist, modes):
    """Return each HE1m's captured share of a Gaussian by direct quadrature.

    ``s_m = |2 pi int_0^a r v_m u dr|^2`` with ``v_m = J0(u_m r/a)/(sqrt(pi) a
    |J1(u_m)|)`` and the Gaussian normalised over the whole plane (section 4.4),
    on one 400-node Gauss--Legendre rule over the core: the integrand is smooth
    there, with at most four oscillations, so the rule is exact to rounding.
    """
    gaussian_norm = math.sqrt(2 / (math.pi * waist**2))
    nodes, weights = np.polynomial.legendre.leggauss(400)
    r = 0.5 * RADIUS * (nodes + 1.0)
    w = 0.5 * RADIUS * weights
    values = []
    for zero in jn_zeros(0, modes):
        norm = 1 / (math.sqrt(math.pi) * RADIUS * abs(j1(zero)))
        integrand = r * norm * j0(zero * r / RADIUS) * np.exp(-((r / waist) ** 2))
        values.append((2 * math.pi * gaussian_norm * float(np.sum(w * integrand))) ** 2)
    return np.array(values)


@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7, 1.0])
def test_gaussian_bessel_capture_agrees_with_an_independent_quadrature(port, ratio):
    """The scalar model's HE1m shares against direct Gaussian--Bessel integrals.

    The captured share in the common metric (``n_ref/Re n_eff`` times the
    launched share) at the carrier bin is the overlap squared, mode by mode. At
    w/a = 0.5 the fundamental takes about 0.906 and the next mode 0.092 (SEP 31,
    section 8.1): a check, not a target.
    """
    grid = port.grid
    waist = ratio * RADIUS
    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, waist),
        space=RadialGrid(8 * waist, 256),
        medium=MEDIUM,
    )
    result = couple(plane, port, model=SCALAR)
    k_bin = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    profiles = port.profiles
    assert profiles is not None
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    captured = (
        weights[k_bin]
        * port.reference_index[k_bin]
        / profiles.effective_index[:, k_bin]
        * np.abs(result.launch.samples[:, 0, k_bin]) ** 2
        / result.report.incident[k_bin]
    )
    expected = bessel_shares(waist, port.shape[0])
    # quad at 1e-13 relative, and the plane resolves the beam to rounding.
    np.testing.assert_allclose(captured, expected, rtol=0.0, atol=1e-11)
    if ratio == 0.5:
        assert captured[0] == pytest.approx(0.906, abs=1e-3)
        assert captured[1] == pytest.approx(0.092, abs=1e-3)
    assert result.report.budget <= SCALAR_BUDGET


# ------------------------------------------------------------------- refinement


def test_nested_mode_sets_domain_and_quadrature_converge_separately():
    """Modal span, transverse window and node count are refined one at a time.

    Orthogonal profiles make each mode's share independent of the set it sits in:
    a larger nested set adds shares and changes none (to rounding). Doubling the
    window at the same spacing changes the shares by the domain check's size, and
    doubling the nodes in the same window by the resolution check's.
    """
    ports = [capillary(modes=count).input_port() for count in (2, 4, 6)]
    grid = ports[0].grid
    waist = 0.6 * RADIUS
    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, waist),
        space=RadialGrid(6 * waist, 192),
        medium=MEDIUM,
    )
    sets = [shares(couple(plane, one, model=SCALAR), grid) for one in ports]
    for smaller, larger in itertools.pairwise(sets):
        np.testing.assert_allclose(larger[: smaller.size], smaller, rtol=1e-12)
        assert np.sum(larger) > np.sum(smaller)
    wider = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, waist),
        space=RadialGrid(12 * waist, 384),
        medium=MEDIUM,
    )
    finer = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, waist),
        space=RadialGrid(6 * waist, 384),
        medium=MEDIUM,
    )
    analytic = plane.analytic
    assert analytic is not None
    for other in (wider, finer):
        changed = shares(couple(other, ports[1], model=SCALAR), grid)
        # The window holds the beam to exp(-72) in intensity and its nodes resolve
        # it, so each refinement moves the shares by rounding of the reductions.
        np.testing.assert_allclose(changed, sets[1], rtol=0.0, atol=1e-12)
    assert analytic.domain_change < 1e-12
    assert analytic.resolution_change < 1e-12


# --------------------------------------------------------------- poor and zero


def test_poor_and_zero_capture_restore_no_energy(port):
    """A wide beam launches its small captured share; zero fields launch zero.

    The uncaptured energy is the incident minus the captured, unclipped and
    nonnegative in the common metric; a y-polarised beam into x-polarised modes is
    an exact zero launch with a zero efficiency, and a zero field a zero launch
    whose efficiency is undefined.
    """
    grid = port.grid
    space = FreeGrid(1.2e-3, 128, 1.2e-3, 128)
    wide = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        cartesian_beam(grid, 4 * RADIUS),
        space=space,
        medium=MEDIUM,
    )
    result = couple(wide, port, model=CouplingModel("scalar_paraxial", model_error=0.1))
    report = result.report
    assert report.efficiency is not None and report.efficiency < 0.15
    uncaptured = report.uncaptured
    assert uncaptured is not None and report.captured is not None
    np.testing.assert_array_equal(uncaptured, report.incident - report.captured)
    assert np.all(uncaptured >= -1e-12 * np.max(report.incident))
    assert report.launched_energy < report.incident_energy
    crossed = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        cartesian_beam(grid, 0.6 * RADIUS, jones=(0, 1)),
        space=FreeGrid(300e-6, 64, 300e-6, 64),
        medium=MEDIUM,
    )
    zero = couple(crossed, port, model=SCALAR)
    assert np.all(zero.launch.samples == 0.0)
    assert zero.report.efficiency == 0.0
    assert zero.report.incident_energy > 0.0

    def nothing(x, y, omega):
        return np.zeros((x.size, omega.size))

    empty = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        nothing,
        space=FreeGrid(300e-6, 32, 300e-6, 32),
        medium=MEDIUM,
    )
    blank = couple(empty, port, model=SCALAR)
    assert np.all(blank.launch.samples == 0.0)
    assert blank.report.efficiency is None
    assert "undefined" in str(blank.report)


# ------------------------------------------------------- bases, gauge, coherence


def handmade_port(modes, grid):
    """Return a modal port of hand-chosen capillary modes, admitting scalar fits."""
    profiles = ScalarModeBasis.from_modes(modes, grid)
    return InputPort(
        "modal",
        grid,
        PlaneBasis("modal", basis_identity("modal", {"labels": list(profiles.labels)})),
        (len(modes), 1, grid.n_omega),
        PlaneGeometry(0.0),
        PlaneReference(),
        PlaneMetric("identity"),
        profiles.core_index,
        models=("same_basis", "scalar_paraxial"),
        profiles=profiles,
    )


def test_a_rotated_degenerate_basis_shares_the_same_total(port):
    """An x-polarised beam enters x alone, or splits evenly over a rotated pair.

    The launched total is the same in either basis to rounding: what is fitted is
    the field.
    """
    grid = port.grid
    straight = [
        MarcatiliMode(RADIUS, "Ar", PRESSURE, phi=phi) for phi in (0.0, math.pi / 2)
    ]
    rotated = [
        MarcatiliMode(RADIUS, "Ar", PRESSURE, phi=phi)
        for phi in (math.pi / 4, 3 * math.pi / 4)
    ]
    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        cartesian_beam(grid, 0.6 * RADIUS),
        space=FreeGrid(300e-6, 64, 300e-6, 64),
        medium=MEDIUM,
    )
    first = couple(plane, handmade_port(straight, grid), model=SCALAR)
    second = couple(plane, handmade_port(rotated, grid), model=SCALAR)
    a, b = shares(first, grid), shares(second, grid)
    # The crossed polarisation's share is the Cartesian sampling's rounding squared.
    assert a[1] < 1e-30
    assert b[0] == pytest.approx(b[1], rel=1e-12)
    assert np.sum(b) == pytest.approx(np.sum(a), rel=1e-12)


def modal_plane(simulation, coefficients):
    """Return a modal plane of given coefficients with a simulation's profiles."""
    grid = simulation.grid
    profiles = ScalarModeBasis.from_modes(simulation.modes, grid)
    samples = np.zeros((profiles.n_modes, 1, grid.n_omega), dtype=complex)
    envelope = np.sqrt(spectral(grid, grid.omega) / 1e-9) * grid.active_mask
    samples[:, 0, :] = np.asarray(coefficients)[:, None] * envelope[None, :]
    return FieldPlane.from_samples(
        samples,
        grid,
        PlaneGeometry(0.0),
        exterior="zero",
        basis=PlaneBasis("modal", "0" * 64, profiles.labels),
        profiles=profiles,
    )


def test_coherent_modes_interfere_and_a_gauge_change_leaves_the_field(port):
    """Two coherent source modes add as fields; flipping a profile's sign does not.

    The launch is linear in the source coefficients, so the two-mode launch is the
    sum of the single-mode ones and its energy is not the sum of theirs; a source
    profile set whose Jones vector and coefficient are both negated is the same
    physical field and launches identically.
    """
    source = capillary(radius=60e-6, modes=2)
    target = capillary(radius=RADIUS, modes=4).input_port()
    both = couple(modal_plane(source, [1.0, 0.7]), target, model=SCALAR)
    first = couple(modal_plane(source, [1.0, 0.0]), target, model=SCALAR)
    second = couple(modal_plane(source, [0.0, 0.7]), target, model=SCALAR)
    np.testing.assert_allclose(
        both.launch.samples,
        first.launch.samples + second.launch.samples,
        rtol=0.0,
        atol=1e-12 * np.max(np.abs(both.launch.samples)),
    )
    # Orthogonal source modes keep their total; the target fundamental's share is
    # where they interfere, and a sign of the second mode moves it.
    grid = target.grid
    fundamental = shares(both, grid)[0]
    separate = shares(first, grid)[0] + shares(second, grid)[0]
    assert abs(fundamental - separate) > 1e-3 * separate
    opposite = couple(modal_plane(source, [1.0, -0.7]), target, model=SCALAR)
    assert shares(opposite, grid)[0] != pytest.approx(fundamental, rel=1e-3)
    plane = modal_plane(source, [1.0, 0.7])
    profiles = cast_profiles(plane)
    flipped = dataclasses.replace(
        profiles, jones=profiles.jones * np.array([[-1.0], [1.0]])
    )
    negated = np.array(plane.samples)
    negated[0] *= -1.0
    gauged = dataclasses.replace(plane, samples=negated, profiles=flipped)
    np.testing.assert_allclose(
        couple(gauged, target, model=SCALAR).launch.samples,
        both.launch.samples,
        rtol=0.0,
        atol=1e-13 * np.max(np.abs(both.launch.samples)),
    )


def cast_profiles(plane):
    """Return a plane's profiles, asserting it has them."""
    assert plane.profiles is not None
    return plane.profiles


def test_a_declared_basis_change_relabels_without_changing_energy(port):
    """``same_basis`` relabels with a unitary change and refuses a non-unitary one."""
    grid = port.grid
    source = capillary(modes=2)
    plane = modal_plane(source, [1.0, 0.5])
    target_identity = basis_identity("modal", {"rotated": True})
    target = dataclasses.replace(
        source.input_port(),
        basis=PlaneBasis("modal", target_identity),
    )
    angle = 0.3
    unitary = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    ) * np.exp(0.2j)
    model = CouplingModel(
        "same_basis",
        basis_change=BasisChange(unitary, plane.basis.identity, target_identity),
    )
    result = couple(plane, target, model=model)
    assert result.report.launched_energy == pytest.approx(
        result.report.incident_energy, rel=1e-13
    )
    np.testing.assert_allclose(
        result.launch.samples[:, 0, :], unitary @ plane.samples[:, 0, :], rtol=1e-14
    )
    stretched = CouplingModel(
        "same_basis",
        basis_change=BasisChange(1.01 * unitary, plane.basis.identity, target_identity),
    )
    with pytest.raises(CouplingError, match="does not preserve energy"):
        couple(plane, target, model=stretched)
    other = CouplingModel(
        "same_basis",
        basis_change=BasisChange(unitary, "1" * 64, target_identity),
    )
    with pytest.raises(CouplingError, match="another pair of bases"):
        couple(plane, target, model=other)
    del grid


def test_a_declared_change_per_bin_relabels_and_one_of_another_shape_is_refused(
    port,
):
    """A frequency-dependent gauge is applied bin by bin; a mismatched one refuses.

    A gauge ``exp(i phi(omega))`` on each mode is a relabelling at every bin, so the
    launch is the plane's times it and the energy is unchanged; the exact report
    names no budget. A change of the wrong size, a change into the wrong number of
    bins, and a change from a spatial plane, whose metric is not modal, are refused.
    """
    grid = port.grid
    source = capillary(modes=2)
    plane = modal_plane(source, [1.0, 0.5])
    target_identity = basis_identity("modal", {"gauged": True})
    target = dataclasses.replace(
        source.input_port(), basis=PlaneBasis("modal", target_identity)
    )
    phase = np.exp(1j * np.linspace(-1.0, 1.0, grid.n_omega))
    gauge = np.zeros((2, 2, grid.n_omega), dtype=complex)
    gauge[0, 0], gauge[1, 1] = phase, phase.conj()

    def declared(matrix, source_identity=plane.basis.identity):
        return CouplingModel(
            "same_basis",
            basis_change=BasisChange(matrix, source_identity, target_identity),
        )

    result = couple(plane, target, model=declared(gauge))
    np.testing.assert_allclose(
        result.launch.samples[:, 0, :],
        np.einsum("tsw,sw->tw", gauge, plane.samples[:, 0, :]),
        rtol=1e-14,
    )
    assert result.report.launched_energy == pytest.approx(
        result.report.incident_energy, rel=1e-13
    )
    text = str(result.report)
    assert "same_basis" in text and "budget" not in text
    with pytest.raises(CouplingError, match="maps 3 columns to 3"):
        couple(plane, target, model=declared(np.eye(3, dtype=complex)))
    with pytest.raises(CouplingError, match=rf"has 5 bins, not {grid.n_omega}"):
        couple(plane, target, model=declared(gauge[:, :, :5]))
    beam = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, 0.5 * RADIUS),
        space=RadialGrid(300e-6, 64),
        medium=MEDIUM,
    )
    columns = np.zeros((4, 64), dtype=complex)
    columns[:, :4] = np.eye(4)
    relabel = CouplingModel(
        "same_basis",
        basis_change=BasisChange(columns, beam.basis.identity, port.basis.identity),
    )
    with pytest.raises(CouplingError, match="identity or dense, got admittance"):
        couple(beam, port, model=relabel)


def test_an_exact_copy_and_a_report_refuse_inconsistent_shapes(port):
    """A plane of the port's basis but not its shape is no copy; a report's per-bin
    arrays are finite."""
    grid = port.grid
    short = FieldPlane.from_samples(
        np.zeros((3, 1, grid.n_omega), dtype=complex),
        grid,
        PlaneGeometry(0.0),
        exterior="zero",
        basis=port.basis,
    )
    with pytest.raises(CouplingError, match=r"shape \(3, 1, \d+\) is not the port's"):
        couple(short, port, model=SAME)
    beam = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, 0.5 * RADIUS),
        space=RadialGrid(300e-6, 128),
        medium=MEDIUM,
    )
    report = couple(beam, port, model=SCALAR).report
    spoiled = np.array(report.incident)
    spoiled[0] = np.nan
    with pytest.raises(CouplingError, match="incident is finite, one value per bin"):
        dataclasses.replace(report, incident=spoiled)


def test_a_spatial_plane_without_an_admittance_is_refused(port):
    """The scalar fit weighs a spatial plane's columns by its admittance."""
    beam = FieldPlane.from_analytic(
        port.grid,
        PlaneGeometry(0.0),
        radial_beam(port.grid, 0.5 * RADIUS),
        space=RadialGrid(300e-6, 64),
        medium=MEDIUM,
    )
    unweighted = dataclasses.replace(beam, metric=PlaneMetric("identity"))
    with pytest.raises(CouplingError, match="an admittance, got identity"):
        couple(unweighted, port, model=SCALAR)


def test_whitening_refuses_a_metric_that_is_not_positive_definite():
    """The extremes are the generalised eigenvalues, and an indefinite metric has no
    whitening."""
    form = np.diag([2.0, 0.5]).astype(complex)
    metric = np.diag([1.0, 2.0]).astype(complex)
    assert _whitened_extremes(form, metric) == pytest.approx((0.25, 2.0), rel=1e-14)
    with pytest.raises(CouplingError, match="not positive definite"):
        _whitened_extremes(form, np.diag([1.0, -1.0]).astype(complex))


# ----------------------------------------------------------- guides and planes


def test_a_capillary_radius_change_by_direct_overlap_and_by_synthesis():
    """A capillary's exit into a wider one: direct overlap against synthesise→project.

    The first capillary runs ten steps; its exit plane read with its simulation
    carries its profiles. Coupled directly, and outcoupled onto a radial and a
    Cartesian plane then coupled, the launches agree to the synthesis window's
    numerical error, and the synthesised plane carries the modal energy to the
    scalar model's metric defect.
    """
    first = capillary(modes=3)
    run = first.run(progress="none")
    plane = field_plane(run, source=first)
    assert plane.profiles is not None
    second = capillary(radius=100e-6, modes=6).input_port()
    direct = couple(plane, second, model=SCALAR)
    assert direct.report.launched_energy < direct.report.incident_energy
    radial = Outcouple(RadialGrid(500e-6, 512)).apply(plane)
    # Matched exit: the field's flux in the core gas against the modal power differs
    # by the profiles' obliquity, n_core cos(theta) against n_eff.
    np.testing.assert_allclose(
        np.sum(radial.synthesised), np.sum(radial.modal), rtol=1e-6
    )
    through = couple(radial.plane, second, model=SCALAR)
    scale = float(np.max(np.abs(direct.launch.samples)))
    assert (
        np.max(np.abs(through.launch.samples - direct.launch.samples))
        <= (10 * radial.window + 1e-12) * scale
    )
    cartesian = Outcouple(FreeGrid(300e-6, 128, 300e-6, 128)).apply(plane)
    across = couple(cartesian.plane, second, model=SCALAR)
    assert (
        np.max(np.abs(across.launch.samples - direct.launch.samples))
        <= (10 * cartesian.window + 1e-12) * scale
    )


def test_the_passive_map_certificate_reports_and_refuses(port):
    """A small apparent gain is a model defect; one beyond the model's defects refuses.

    The capillary map's whitened gain is the effective-index defect, reported in
    the budget and never rescaled; a gain larger than every metric, angular and
    closure defect together is refused, never clipped.
    """
    grid = port.grid
    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, 0.5 * RADIUS),
        space=RadialGrid(300e-6, 192),
        medium=MEDIUM,
    )
    result = couple(plane, port, model=SCALAR)
    passivity = next(item for item in result.report.items if item.name == "passivity")
    metric = next(item for item in result.report.items if item.name == "target metric")
    assert passivity.value is not None and metric.value is not None
    assert passivity.value <= metric.value + 1e-3
    with pytest.raises(CouplingError, match="apparent gain"):
        _gain_item(0.5, [])
    assert _gain_item(-0.1, []).value == 0.0


def step_index(radius, contrast, wavelength=1064e-9, core=1.45, grid=None):
    """Return a GNLSE simulation of an HE11 step-index fibre, by default at 1064 nm."""
    cladding = core * math.sqrt(1 - 2 * contrast)
    fibre = StepIndexMode(
        radius, core=core, cladding=cladding, profile_wavelength=wavelength
    )
    grid = (
        {
            "wavelength": wavelength,
            "wavelength_limits": (900e-9, 1300e-9),
            "trange": 400e-15,
        }
        if grid is None
        else grid
    )
    return Simulation.from_gnlse(
        None,
        1e-3,
        fibre=fibre,
        n2=2.6e-20,
        fwhm=100e-15,
        energy=1e-12,
        n_save=2,
        **{"noise": None, **grid},
    )


def test_a_weakly_guiding_fibre_pair_is_admitted_within_its_envelope():
    """One weak-guidance step-index pair: the LP01 overlap, the screen and the envelope.

    Two fibres of contrast 0.005 and core radii in ratio 0.85: the launched share is
    the LP01 overlap squared times the metric ratio, the screen passes at a 0.1
    budget and is replaced by the envelope where it does not, and a contrast beyond
    the envelope's range is refused.
    """
    wavelength = 1064e-9
    aperture = math.sqrt(1.45**2 * 2 * 0.005)
    target_radius = 2.2 / (2 * math.pi / wavelength * aperture)
    source = step_index(0.85 * target_radius, 0.005)
    target = step_index(target_radius, 0.005).input_port()
    assert target.models == ("same_basis", "scalar_paraxial")
    assert target.contrast is not None
    plane = modal_plane(source, [1.0])
    result = couple(
        plane, target, model=CouplingModel("scalar_paraxial", model_error=0.1)
    )
    names = {item.name: item for item in result.report.items}
    assert names["contrast"].value == pytest.approx(0.005, rel=1e-9)
    assert names["envelope"].value is not None
    profiles_source = cast_profiles(plane)
    profiles_target = target.profiles
    assert profiles_target is not None
    r, w = radial_rule(
        [profiles_source.radius, profiles_target.radius],
        max(profiles_source.outer(), profiles_target.outer()),
        panel=profiles_source.radius / 8,
    )
    overlap = float(
        np.sum(w * profiles_source.values(r)[0] * profiles_target.values(r)[0])
    )
    captured = result.report.captured
    assert captured is not None
    assert np.sum(captured) / result.report.incident_energy == pytest.approx(
        overlap**2, rel=1e-12
    )
    # At a budget whose eighth is below the contrast, the recorded envelope stands in
    # for the screen (the pair's angular content then decides the total).
    occupied = np.asarray(target.reference_index) > 0
    replaced = _contrast_items(
        profiles_target,
        plane,
        np.asarray(target.reference_index),
        occupied,
        0.02,
        {"modes": 1.0, "radius_ratio": 0.85, "v_source": 1.87, "v_target": 2.2},
    )
    assert "replaced by" in replaced[0].note
    assert replaced[0].passed
    strong = step_index(target_radius, 0.02).input_port()
    with pytest.raises(CouplingError, match="contrast"):
        couple(
            modal_plane(step_index(0.85 * target_radius, 0.02), [1.0]),
            strong,
            model=CouplingModel("scalar_paraxial", model_error=0.5),
        )


def test_a_beam_into_a_weakly_guiding_fibre_is_admitted_within_its_envelope():
    """A Gaussian beam into a step-index fibre: the LP01 overlap and its envelope.

    A fibre of contrast 0.005 and V 2.2 at 1064 nm, the beam sampled in the core
    index: the envelope item is the beam-into-fibre allowance at the fibre's
    contrast, the screen passes at a 0.1 budget and is replaced by the envelope
    where it does not, and a contrast or a V beyond the envelope's range is refused
    by name.
    """
    wavelength = 1064e-9

    def radius_for(v, contrast):
        return v / (2 * math.pi / wavelength * math.sqrt(1.45**2 * 2 * contrast))

    def beam_into(port, waist):
        return FieldPlane.from_analytic(
            port.grid,
            PlaneGeometry(0.0),
            radial_beam(port.grid, waist),
            space=RadialGrid(40 * waist, 256),
            index=np.asarray(port.reference_index),
        )

    radius = radius_for(2.2, 0.005)
    target = step_index(radius, 0.005).input_port()
    plane = beam_into(target, 1.2 * radius)
    report = couple(
        plane, target, model=CouplingModel("scalar_paraxial", model_error=0.1)
    ).report
    names = {item.name: item for item in report.items}
    assert names["contrast"].value == pytest.approx(0.005, rel=1e-9)
    assert names["contrast"].passed
    # The contrast-scaled allowance, plus the incident share of the blue bins
    # where the fibre's V exceeds 2.4 (2.4e-5 of this 30 fs pulse).
    v = target.grid.omega * radius * math.sqrt(1.45**2 * 2 * 0.005) / C
    beyond = (report.incident > 0) & (v > 2.4)
    share = float(np.sum(report.incident[beyond]) / report.incident_energy)
    assert share > 1e-6
    assert names["envelope"].value == pytest.approx(
        STEP_INDEX_BEAM_ENVELOPE.allowance({"contrast": 0.005}) + share, rel=1e-9
    )
    assert names["envelope"].note.startswith(STEP_INDEX_BEAM_ENVELOPE.identity)
    assert report.efficiency is not None and 0.95 < report.efficiency < 1.0
    profiles = target.profiles
    assert profiles is not None
    occupied = np.asarray(target.reference_index) > 0
    replaced = _contrast_items(
        profiles,
        plane,
        np.asarray(target.reference_index),
        occupied,
        0.02,
        {"modes": 1.0, "v_target": 2.2},
    )
    assert f"replaced by {STEP_INDEX_BEAM_ENVELOPE.identity}" in replaced[0].note
    assert replaced[1].value == pytest.approx(5e-3, rel=1e-9)
    strong = step_index(radius_for(2.2, 0.02), 0.02).input_port()
    with pytest.raises(
        CouplingError,
        match=r"contrast = 0\.02\d* lies outside the envelope step-index-beam",
    ):
        couple(
            beam_into(strong, radius_for(2.2, 0.02)),
            strong,
            model=CouplingModel("scalar_paraxial", model_error=0.5),
        )
    thick = step_index(radius_for(2.6, 0.005), 0.005).input_port()
    with pytest.raises(CouplingError, match=r"v_target = 2\.(6|59)\d* lies outside"):
        couple(
            beam_into(thick, radius_for(2.6, 0.005)),
            thick,
            model=CouplingModel("scalar_paraxial", model_error=0.5),
        )


# --------------------------------------------------------------------- refusals


def test_couple_refuses_by_name(port):
    """Every unqualified pair, missing budget and mismatch is refused by name."""
    grid = port.grid
    with pytest.raises(CouplingError, match="takes no model_error"):
        CouplingModel("same_basis", model_error=0.1)
    with pytest.raises(CouplingError, match="give its energy budget"):
        CouplingModel("scalar_paraxial")
    with pytest.raises(CouplingError, match=r"lies in \(0, 1\)"):
        CouplingModel("vector_paraxial", model_error=1.0)
    unitary = BasisChange(np.eye(1, dtype=complex), "0" * 64, "0" * 64)
    with pytest.raises(CouplingError, match="belongs to same_basis"):
        CouplingModel("scalar_paraxial", model_error=0.1, basis_change=unitary)
    with pytest.raises(CouplingError, match="finite"):
        BasisChange(np.full((1, 1), np.nan, dtype=complex), "0" * 64, "0" * 64)
    beam = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, 0.5 * RADIUS),
        space=RadialGrid(300e-6, 128),
        medium=MEDIUM,
    )
    with pytest.raises(CouplingError, match="not vector_paraxial"):
        couple(beam, port, model=CouplingModel("vector_paraxial", model_error=0.1))
    with pytest.raises(CouplingError, match="a coupling, not a relaunch"):
        couple(beam, port, model=SAME)
    with pytest.raises(CouplingError, match="sum to"):
        couple(beam, port, model=CouplingModel("scalar_paraxial", model_error=3e-3))


def test_couple_refuses_unqualified_pairs_by_name(port):
    """A port without profiles, a plane without them, beyond the envelope, a fibre
    target for a beam beyond its envelope, mixed families, a general plane, another
    orientation or lattice."""
    grid = port.grid
    beam = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, 0.5 * RADIUS),
        space=RadialGrid(300e-6, 128),
        medium=MEDIUM,
    )
    bare = dataclasses.replace(beam, profiles=None)
    profiles = cast_profiles_or_none(port)
    no_profile_port = dataclasses.replace(port, profiles=None)
    with pytest.raises(CouplingError, match="no scalar profiles"):
        couple(bare, no_profile_port, model=SCALAR)
    del profiles
    modal = FieldPlane.from_samples(
        np.zeros((4, 1, grid.n_omega), dtype=complex),
        grid,
        PlaneGeometry(0.0),
        exterior="zero",
        basis=PlaneBasis("modal", "0" * 64),
    )
    with pytest.raises(CouplingError, match="carries no mode profiles"):
        couple(modal, port, model=SCALAR)
    many = capillary(modes=9).input_port()
    with pytest.raises(CouplingError, match="outside the envelope"):
        couple(beam, many, model=SCALAR)
    hybrid = capillary(modes=[MarcatiliMode(RADIUS, "Ar", PRESSURE, n=2)])
    assert hybrid.input_port().models == ("same_basis",)
    fibre = step_index(2.4e-6, 0.005, wavelength=800e-9, grid=GRID).input_port()
    assert fibre.lattice == port.lattice
    # V = 2.7 at 800 nm: beyond the beam-into-fibre envelope's range.
    with pytest.raises(CouplingError, match="v_target"):
        couple(beam, fibre, model=SCALAR)
    with pytest.raises(CouplingError, match="not qualified for a capillary plane"):
        couple(modal_plane(capillary(modes=2), [1.0, 0.0]), fibre, model=SCALAR)
    general = dataclasses.replace(
        modal, kind="general_modal", basis=PlaneBasis("general_modal", "0" * 64)
    )
    with pytest.raises(CouplingError, match="radial, Cartesian or modal"):
        couple(general, port, model=SCALAR)
    shifted = dataclasses.replace(
        beam,
        geometry=PlaneGeometry(0.0, normal=(0.0, 1.0, 0.0), x_axis=(1.0, 0.0, 0.0)),
    )
    with pytest.raises(CouplingError, match="oriented differently"):
        couple(shifted, port, model=SCALAR)
    other = capillary()
    longer = Simulation.from_capillary(
        RADIUS,
        1e-4,
        "Ar",
        PRESSURE,
        modes=4,
        n_save=2,
        noise=None,
        envelope=True,
        energy=1e-6,
        fwhm=30e-15,
        wavelength=800e-9,
        wavelength_limits=(700e-9, 920e-9),
        trange=400e-15,
    )
    with pytest.raises(CouplingError, match="regrid it first"):
        couple(beam, longer.input_port(), model=SCALAR)
    del other


def cast_profiles_or_none(port):
    """Return a port's profiles, which may be None."""
    return port.profiles


def test_from_analytic_records_and_refuses(port):
    """The analytic constructor's checks and its refusals."""
    grid = port.grid
    space = RadialGrid(300e-6, 64)
    index = np.where(grid.omega > 0, 1.0, 0.0)
    supported = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, 30e-6),
        space=space,
        index=index,
        support=200e-6,
    )
    assert supported.analytic is not None
    assert supported.analytic.support == 200e-6
    assert supported.analytic.error == max(
        supported.analytic.domain_change, supported.analytic.resolution_change
    )
    with pytest.raises(CouplingError, match="one"):
        FieldPlane.from_analytic(
            grid, PlaneGeometry(0.0), radial_beam(grid, 30e-6), space=space
        )
    with pytest.raises(CouplingError, match="outside the window"):
        FieldPlane.from_analytic(
            grid,
            PlaneGeometry(0.0),
            radial_beam(grid, 30e-6),
            space=space,
            index=index,
            support=1.0,
        )

    def wrong(r, omega):
        return np.zeros((r.size, 3, omega.size))

    with pytest.raises(CouplingError, match="n_pol 1 or 2"):
        FieldPlane.from_analytic(
            grid, PlaneGeometry(0.0), wrong, space=space, index=index
        )

    def broken(r, omega):
        return np.full((r.size, omega.size), np.nan)

    with pytest.raises(CouplingError, match="nonfinite"):
        FieldPlane.from_analytic(
            grid, PlaneGeometry(0.0), broken, space=space, index=index
        )
    with pytest.raises(CouplingError, match="reference index"):
        PlaneMetric.paraxial(space, -index, grid)
    with pytest.raises(CouplingError, match="takes a SpatialPulse"):
        FieldPlane.from_pulse(object(), grid, PlaneGeometry(0.0), space=space)
    beam = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 3e-5
    )
    with pytest.raises(CouplingError, match="medium= or an index="):
        FieldPlane.from_pulse(beam, grid, PlaneGeometry(0.0), space=space)
    with pytest.raises(CouplingError, match="propz=0"):
        FieldPlane.from_pulse(
            SpatialPulse(beam.temporal, 3e-5, propz=1e-3),
            grid,
            PlaneGeometry(0.0),
            space=space,
            index=index,
        )
    measured = FieldPlane.from_pulse(
        beam, grid, PlaneGeometry(0.0), space=space, index=index
    )
    # The incident measure M0 holds the request; the axial flux is below it by the
    # beam's obliquity, a few parts in 1e5 for a 30 um beam at 800 nm.
    assert measured.energy() == pytest.approx(1e-6, rel=1e-4)
    assert measured.energy() < 1e-6


def test_outcouple_refuses_what_it_does_not_model(port):
    """Only a modal plane with profiles, into a matched medium, under 'matched'."""
    grid = port.grid
    plane = modal_plane(capillary(modes=2), [1.0, 0.0])
    space = RadialGrid(300e-6, 128)
    with pytest.raises(CouplingError, match="not both"):
        Outcouple(space, medium=MEDIUM, index=np.ones(grid.n_omega))
    with pytest.raises(CouplingError, match="core index"):
        Outcouple(space, medium=SpatialMedium()).apply(plane)
    with pytest.raises(CouplingError, match="with its profiles"):
        Outcouple(space).apply(dataclasses.replace(plane, profiles=None))
    matched = Outcouple(space, medium=MEDIUM).apply(plane)
    assert matched.plane.kind == "radial"
    assert matched.plane.provenance.history[-1].startswith("Outcouple(matched)")
    assert repr(Outcouple(space)) == "Outcouple(matched, RadialGrid)"


def test_the_report_prints_and_carries_its_terms(port):
    """The report's block, its budget and its admission items."""
    grid = port.grid
    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        radial_beam(grid, 0.5 * RADIUS),
        space=RadialGrid(300e-6, 128),
        medium=MEDIUM,
    )
    report = couple(plane, port, model=SCALAR).report
    names = [item.name for item in report.items]
    for name in (
        "angular",
        "numerical",
        "target metric",
        "source metric",
        "contrast",
        "envelope",
        "passivity",
    ):
        assert name in names
    assert report.budget == pytest.approx(
        sum(item.value for item in report.items if item.counted and item.value)
    )
    assert report.budget <= SCALAR_BUDGET
    assert all(item.passed for item in report.items)
    text = str(report)
    assert "budget" in text and "uncaptured" in text
    assert "array(" not in repr(report)
    angle = cone_angle(0.05)
    assert 1 - math.cos(angle) == pytest.approx(0.05 / 8, rel=1e-14)
    assert EPS0 > 0


def test_a_source_path_certificate_is_carried_beside_the_coupling(port):
    """A gradient run's rebuilt certificate reaches the report, separately."""
    grid = FieldGrid.from_limits(2e-3, 800e-9, (600e-9, 1.2e-6), 400e-15)
    gradient = Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 32),
        SpatialMedium("Ar", PressureGradient.two_point(2e-3, 1.0, 2.0)),
        [],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 6e-5),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[2e-3],
            dz_max=2e-4,
            dz_init=2e-4,
            backend="numpy_ref",
        ),
        noise=None,
    )
    flux = gradient.translate().flux
    assert flux is not None and flux.admission is not None
    plane = dataclasses.replace(
        FieldPlane.from_analytic(
            port.grid,
            PlaneGeometry(0.0),
            radial_beam(port.grid, 0.5 * RADIUS),
            space=RadialGrid(400e-6, 128),
            medium=MEDIUM,
        ),
        certificate=flux.admission,
    )
    report = couple(plane, port, model=SCALAR).report
    assert report.certificate is flux.admission
    assert "path certificate" in str(report)
    plain = couple(dataclasses.replace(plane, certificate=None), port, model=SCALAR)
    assert plain.report.certificate is None
    assert "path certificate" not in str(plain.report)


def test_a_displaced_tilted_beam_launches_with_the_overlap_s_phase(port):
    """The launch's phase is that of the overlap, pinning the transform's sign.

    Every share is invariant under conjugating the field, and on every FreeGrid the
    origin phase is exp(-i pi m) for either sign, so neither catches a sign error;
    the phase of a coefficient does. A beam displaced by a distance that is not a
    multiple of the spacing and tilted has complex overlaps
    ``int J0(u_m r/a) E dA`` (optics convention), here by an independent polar
    Gauss--Legendre rule; the optics-convention launch b_m = sqrt(n_eff) a_m has
    the same phase, mode by mode.
    """
    grid = port.grid
    shift, tilt = 0.13 * RADIUS, 1.5e-3
    beam = cartesian_beam(grid, 0.5 * RADIUS, shift=shift, tilt=tilt)
    plane = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        beam,
        space=FreeGrid(4 * RADIUS, 128, 4 * RADIUS, 128),
        medium=MEDIUM,
    )
    k_bin = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    launched = couple(plane, port, model=SCALAR).launch.samples[:, 0, k_bin]
    nodes, weights = np.polynomial.legendre.leggauss(300)
    radii = 0.5 * RADIUS * (nodes + 1)
    n_theta = 128
    theta = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    r = np.repeat(radii, n_theta)
    t = np.tile(theta, radii.size)
    area = np.repeat(0.5 * RADIUS * weights * radii, n_theta) * 2 * np.pi / n_theta
    field = beam(r * np.cos(t), r * np.sin(t), grid.omega[k_bin : k_bin + 1])[:, 0, 0]
    overlaps = np.array(
        [np.sum(area * j0(u * r / RADIUS) * field) for u in jn_zeros(0, 4)]
    )
    assert np.all(np.abs(np.angle(overlaps)) > 0.05)  # genuinely complex
    # Matched computations of one integral by two rules: measured 2e-14 rad.
    np.testing.assert_allclose(np.angle(launched / overlaps), 0.0, atol=1e-9)


def test_band_bins_outside_an_envelope_s_range_are_counted_in_full():
    """V across the band, and a capillary's leaky parameter, beyond the envelope.

    The step-index profiles are frozen at 1064 nm, but V grows with frequency: the
    bins where either fibre's V leaves the envelope's range are not bounded by it,
    so their incident share is added to the envelope item. A capillary bin whose
    leaky parameter exceeds the range is counted the same way.
    """
    wavelength = 1064e-9
    aperture = math.sqrt(1.45**2 * 2 * 0.005)
    target_radius = 2.35 / (2 * math.pi / wavelength * aperture)
    source = step_index(0.95 * target_radius, 0.005)
    target = step_index(target_radius, 0.005).input_port()
    plane = modal_plane(source, [1.0])
    # A budget that holds the uncovered share (about a quarter of the energy).
    report = couple(
        plane, target, model=CouplingModel("scalar_paraxial", model_error=0.5)
    ).report
    grid = target.grid
    v = grid.omega * target_radius * aperture / C
    beyond = (report.incident > 0) & (v > 2.4)
    share = float(np.sum(report.incident[beyond]) / report.incident_energy)
    assert share > 1e-3  # the blue side of the band: V above 2.4
    item = next(item for item in report.items if item.name == "envelope")
    # The same weighted sum in another order (the indices are constant here).
    assert item.value == pytest.approx(STEP_INDEX_ENVELOPE.bound + share, rel=1e-9)
    assert "outside its range" in item.note
    narrow = capillary(radius=21e-6, modes=1).input_port()
    wide = FieldPlane.from_analytic(
        narrow.grid,
        PlaneGeometry(0.0),
        radial_beam(narrow.grid, 12e-6),
        space=RadialGrid(150e-6, 128),
        medium=MEDIUM,
    )
    leaky_report = couple(
        wide, narrow, model=CouplingModel("scalar_paraxial", model_error=0.2)
    ).report
    leaky_item = next(item for item in leaky_report.items if item.name == "envelope")
    assert "outside its range" in leaky_item.note

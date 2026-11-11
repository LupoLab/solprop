# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The fluence widths the native backend reduces in place, against the reference.

A separable route's fluence takes no time transform natively: Parseval's theorem turns
the reference's sum over ``n_t`` samples into a sum over the bins it has already
synthesised. On a Cartesian plane the centroid, the second moment and the azimuthal
average are reductions too, so only their numbers cross. The reference forms each of
these on the host, in time. Every comparison is one state, loaded at one plane, held to
a rounding budget fixed here before the comparison (SEP 25), on the states a width is
most exposed on: a beam off centre, a hollow profile, one node, one column, one bin, a
beam in a corner, and two beams that cancel on the axis.
"""

import importlib.util
import math
import platform

import numpy as np
import pytest

import solprop.backends.spatial as host_spatial
from backends._session_probe_b2 import launch_statistics, launched
from backends.planar_cases import planar_case
from bounds import assert_within, device_backend, gamma
from solprop.backends import select
from solprop.backends.events import Terminal
from solprop.backends.session import Start
from solprop.backends.spatial import fluence_budget, plane_fluence_widths
from solprop.constants import EPS0, C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BackendError, GridError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import (
    SpatialMedium,
    _fluence_rings,
    _ring_fluence_widths,
    cartesian_fluence_widths,
)
from solprop.stats import StatsSpec
from solprop.stepper import PIController, RungeKutta

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

WIDTHS = ("beam_radius", "fwhm_r")
LOSSY = SpatialMedium(1.45 + 2e-3j)
"""Glass whose admittance is complex, so that the closure's cross terms are real."""

HAS_NATIVE = importlib.util.find_spec("solprop_native") is not None
native_only = pytest.mark.skipif(not HAS_NATIVE, reason="solprop-native is absent")


def _time_grid(kind):
    if kind == "envelope":
        return EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    return FieldGrid.from_limits(1e-3, 800e-9, (400e-9, 2e-6), 40e-15)


def _simulation(kind, space, n_pol, medium=LOSSY, waist=60e-6):
    grid = _time_grid(kind)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[1e-3],
        dz_max=2.5e-4,
        dz_init=2.5e-4,
        absorber=None,
        integrator=RungeKutta(),
        stats=StatsSpec(names=WIDTHS),
    )
    temporal = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9)
    pulse = (
        SpatialPulse(temporal, waist, jones=(1, 0.5j))
        if n_pol == 2
        else SpatialPulse(temporal, waist)
    )
    simulation = Simulation.from_free(
        grid, space, medium, [], pulse, settings=settings, noise=None
    )
    return grid, simulation.translate(), simulation._run_request()


def _gaussian(x, y, x0=0.0, y0=0.0, waist=6e-5):
    return np.exp(-(((x - x0) ** 2 + (y - y0) ** 2) / waist**2))


def _profile(name, x, y):
    """Return one state's transverse amplitude, on the grid of ``x`` and ``y``."""
    if name == "off_centre":
        return _gaussian(x, y, 7.3e-5, -4.1e-5)
    if name == "hollow":
        return np.hypot(x, y) / 6e-5 * _gaussian(x, y)
    if name == "one_node":
        return ((x == x.flat[x.size // 2]) & (y == y.flat[y.size // 2])).astype(float)
    if name == "corner":
        return _gaussian(x, y, x.min(), y.min(), 2e-5)
    # A line of near-cancellation through the axis, where the budget is read as zero.
    assert name == "cancelling"
    return _gaussian(x, y, 2e-5, 0.0) - _gaussian(x, y, -2e-5, 0.0)


def _state(problem, grid, name):
    """Return an internal spectrum, with the launch's own temporal shape."""
    geometry = problem.transform.geometry
    launch = np.asarray(problem.initial.field0)
    n_y, n_x = geometry.axes[1].size, geometry.axes[0].size
    n_pol, n_omega = problem.shape[1], problem.shape[2]
    # The launch's own bins, weighted by polarisation, and a chirp-like phase, so that
    # the bins interfere in time and the temporal sum is not trivially the spectral one.
    rng = np.random.default_rng(34)
    spectrum = launch.reshape(n_y * n_x, n_pol, n_omega).sum(axis=0)
    spectrum = spectrum * np.exp(1j * rng.uniform(-0.3, 0.3, n_omega))
    if name == "one_bin":
        peak = int(np.argmax(np.sum(np.abs(spectrum) ** 2, axis=0)))
        bin_only = np.zeros_like(spectrum)
        bin_only[:, peak] = spectrum[:, peak]
        spectrum = bin_only
        x, y = np.meshgrid(geometry.axes[0], geometry.axes[1])
        columns = np.fft.fft2(_gaussian(x, y)).ravel()
    elif name == "one_column":
        columns = np.zeros(n_y * n_x, complex)
        columns[n_x + 1] = 1.0
    else:
        x, y = np.meshgrid(geometry.axes[0], geometry.axes[1])
        columns = np.fft.fft2(_profile(name, x, y)).ravel()
    field = np.einsum("r,pw->rpw", columns, spectrum)
    return field * (1e3 / np.max(np.abs(field)))


def _measure(name, grid, problem, run_request, field, spec):
    request = run_request.for_run(stats=spec)
    backend = select(name, problem, request, grid=grid)
    try:
        return launch_statistics(backend, problem, request, field)
    finally:
        backend.close()


def _load(backend, field):
    """Load `field` into the native state directly, for its kernels' own calls."""
    flat = np.ascontiguousarray(field, dtype=np.complex128).reshape(-1)
    state = backend._native_state  # pyright: ignore[reportAttributeAccessIssue]
    state.set_initial(flat, np.empty(0, dtype=np.complex128))
    return state


def _width_budget(grid, problem):
    """A width's rounding budget, relative to its value (``test_flux_parity._budget``).

    The fluence synthesises E and G at the nodes (``n_columns`` terms, a transverse
    FFT) and the reference inverts the time transform (log2 n_t stages); the pairing
    sums ``n_pol * n_t`` products. A width interpolates a fluence whose relative slope
    is of order one at its crossing, which a factor of eight covers.
    """
    n_columns, n_pol = problem.shape[0], problem.shape[1]
    return gamma(8 * 16 * (n_columns + n_pol * grid.n_t + 64 * int(np.log2(grid.n_t))))


def _agree(got, want, bound, pixel, what, extent):
    for name in WIDTHS:
        actual, expected = np.ravel(got[name]), np.ravel(want[name])
        # A width the rule leaves undefined (a hollow profile, a beam at an edge) is
        # undefined on both, an exact structural contract.
        np.testing.assert_array_equal(
            np.isnan(actual), np.isnan(expected), err_msg=what
        )
        live = ~np.isnan(expected)
        if np.any(live):
            # A pixel at least: a point beam's moment is its centroid's rounding, whose
            # size is the axes', not the width's.
            scale = max(float(np.linalg.norm(expected[live])), pixel)
            allowed = bound
            if name == "beam_radius" and float(np.max(expected[live])) < pixel:
                # A moment errs by the budget relative to the axes' squares, and a
                # width is its root: a point beam's has no size of its own to hide that.
                allowed = 2 * math.sqrt(bound) * extent / scale
            assert_within(
                actual[live],
                expected[live],
                allowed,
                scale=scale,
                what=f"{what} {name}",
            )


STATES = (
    "off_centre",
    "hollow",
    "one_node",
    "one_column",
    "one_bin",
    "corner",
    "cancelling",
)


def _check_plane(kind, n_pol, state):
    # Unequal samples and spacings: the rule's dx, dy and its y-first order.
    space = FreeGrid(3e-4, 32, 4e-4, 24)
    grid, problem, run_request = _simulation(kind, space, n_pol)
    assert problem.shape[1] == n_pol
    field = _state(problem, grid, state)
    spec = StatsSpec(names=(*WIDTHS, "on_axis_intensity", "peak_intensity"))
    want = _measure("numpy_ref", grid, problem, run_request, field, spec)
    got = _measure("native", grid, problem, run_request, field, spec)
    geometry = problem.transform.geometry
    assert geometry is not None
    x = geometry.axes[0]
    pixel = float(x[1] - x[0])
    what = f"{kind} {n_pol} {state}"
    extent = max(float(np.max(np.abs(axis))) for axis in geometry.axes)
    _agree(got, want, _width_budget(grid, problem), pixel, what, extent)
    # The axis's coherent sum errs normwise, as a node's synthesis does, so on the
    # states that cancel on the axis the scale is the beam's peak, not the axis's value.
    # The time transform and the pairing as for the fluence (`_width_budget`), and no
    # crossing to interpolate, so without its factor of eight.
    peak = float(np.max(want["peak_intensity"]))
    for name in ("on_axis_intensity", "peak_intensity"):
        assert_within(
            got[name],
            want[name],
            _width_budget(grid, problem) / 8,
            scale=peak,
            what=f"{what} {name}",
        )


@native_only
@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_a_plane_s_widths_are_the_reference_s(kind, n_pol, state):
    _check_plane(kind, n_pol, state)


def _require_device():
    if device_backend("native") == "none":
        pytest.skip("the installed solprop-native build has no accelerator backend")


@pytest.mark.gpu
@native_only
@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("n_pol", [1, 2])
@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_a_device_s_widths_are_the_reference_s(kind, n_pol, state):
    # The reductions run in the device's memory space; only the numbers cross.
    _require_device()
    _check_plane(kind, n_pol, state)


def _route(route, kind="envelope"):
    if route == "plane":
        return _simulation(kind, FreeGrid(3e-4, 32, 4e-4, 24), 2)
    if route == "plane_scalar":
        return _simulation(kind, FreeGrid(3e-4, 32, 4e-4, 24), 1)
    if route == "line":
        return _simulation(kind, LineGrid(4e-4, 48), 2)
    assert route == "planar"
    # The field grid's time is padded, which a slab's fluence has to scale away.
    grid, _, problem, run_request = planar_case(
        "kerr" if kind == "envelope" else "field-kerr", steps=0
    )
    return grid, problem, run_request.for_run(stats=StatsSpec(names=WIDTHS))


def _field_for(route, grid, problem):
    rng = np.random.default_rng(34)
    field = np.asarray(problem.initial.field0)
    # The bins and the columns interfere at the nodes, so every node's cross terms and
    # every bin's weight enter its fluence.
    field = field * np.exp(1j * rng.uniform(-0.3, 0.3, field.shape[-1]))
    return field * np.exp(1j * rng.uniform(-0.3, 0.3, (field.shape[0], 1, 1)))


def _check_fluence(monkeypatch, route, kind):
    grid, problem, run_request = _route(route, kind)
    field = _field_for(route, grid, problem)
    held = {}
    function = host_spatial.signed_fluence_widths

    def capture(geometry, fluence, budget, axis):
        held.update(fluence=fluence.copy(), budget=budget.copy())
        return function(geometry, fluence, budget, axis)

    monkeypatch.setattr(host_spatial, "signed_fluence_widths", capture)
    _measure("numpy_ref", grid, problem, run_request, field, StatsSpec(names=WIDTHS))
    monkeypatch.undo()
    backend = select("native", problem, run_request, grid=grid)
    try:
        state = _load(backend, field)
        packed = np.ravel(state.spatial_statistics(16, 0.5 * EPS0 * C, 0.0))
    finally:
        backend.close()
    n_nodes = problem.transform.n_nodes
    fluence, axis, largest = packed[:n_nodes], packed[n_nodes], packed[n_nodes + 1]
    # A separable route leaves the axis to a radial width alone.
    assert math.isnan(axis)
    n_terms = problem.shape[1] * grid.n_t + problem.transform.n_modes
    budget = fluence_budget(np.full(n_nodes, largest), n_terms)
    # Cauchy-Schwarz bounds the reference's magnitude by the native one, to the rounding
    # of the two sums.
    assert budget[0] >= held["budget"][0] * (1 - gamma(4 * n_terms))
    # Each fluence errs within its own normwise budget (`fluence_budget`), and the two
    # share the state, so they differ by at most the two budgets.
    assert_within(
        fluence,
        held["fluence"],
        budget + held["budget"],
        what=f"{route} {kind} fluence",
        elementwise=True,
    )
    return budget[0], held["budget"][0]


@native_only
@pytest.mark.parametrize("kind", ["envelope", "field"])
@pytest.mark.parametrize("route", ["plane", "plane_scalar", "line"])
def test_a_separable_fluence_is_the_reference_s_at_every_node(monkeypatch, route, kind):
    native, reference = _check_fluence(monkeypatch, route, kind)
    # A beam whose admittance varies little over its band: the bound is nearly tight,
    # so the refusal is the reference's in all but a narrow band of rounding.
    assert native <= 1.05 * reference


@native_only
@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_a_planar_fluence_is_the_reference_s_at_every_node(monkeypatch, kind):
    # The hybrid kind's guide has real dispersion across its slab orders, so its bound
    # need not be nearly tight. It is only held to bound the reference's from above.
    _check_fluence(monkeypatch, "planar", kind)


@native_only
@pytest.mark.parametrize("kind", ["envelope", "field"])
@pytest.mark.parametrize("route", ["plane", "plane_scalar"])
def test_a_plane_s_moments_bound_the_refusal_as_its_fluence_does(route, kind):
    # What a plane's refusal scales by is in its moments, and its width comes out the
    # same whatever that is unless a negative lobe is near it: flag 16's value is held
    # to the reference's, so flag 32's is held to that.
    grid, problem, run_request = _route(route, kind)
    field = _field_for(route, grid, problem)
    backend = select("native", problem, run_request, grid=grid)
    try:
        state = _load(backend, field)
        fluence = np.ravel(state.spatial_statistics(16, 0.5 * EPS0 * C, 0.0))
        moments = np.ravel(state.spatial_statistics(32, 0.5 * EPS0 * C, 0.0))
    finally:
        backend.close()
    n_nodes = problem.transform.n_nodes
    largest = fluence[n_nodes + 1]
    assert largest > 0.0
    assert moments[2] == pytest.approx(largest, rel=1e-12, abs=0.0)
    # And the fluence's own largest value and most negative one, the refusal's input.
    assert moments[3] == pytest.approx(np.max(fluence[:n_nodes]), rel=1e-12, abs=0.0)
    assert moments[0] == pytest.approx(
        np.min(fluence[:n_nodes]), rel=1e-12, abs=1e-12 * largest
    )
    assert moments[1] == np.argmin(fluence[:n_nodes])


def _check_axis(route):
    """One route's on-axis intensity: the far-field slots', or the planar kernel's."""
    grid, problem, run_request = _route(route)
    field = _field_for(route, grid, problem)
    spec = StatsSpec(names=("on_axis_intensity", "peak_intensity"))
    want = _measure("numpy_ref", grid, problem, run_request, field, spec)
    got = _measure("native", grid, problem, run_request, field, spec)
    # The fluence's budget without the interpolation's factor of eight.
    assert_within(
        got["on_axis_intensity"],
        want["on_axis_intensity"],
        _width_budget(grid, problem) / 8,
        scale=float(np.max(want["peak_intensity"])),
        what=f"{route} axis",
    )


@native_only
@pytest.mark.parametrize("route", ["plane", "plane_scalar", "line", "planar"])
def test_the_on_axis_intensity_is_the_reference_s(route):
    _check_axis(route)


@pytest.mark.gpu
@native_only
@pytest.mark.parametrize("route", ["plane", "plane_scalar", "line", "planar"])
def test_a_device_s_on_axis_intensity_is_the_reference_s(route):
    _require_device()
    _check_axis(route)


@pytest.mark.gpu
@native_only
@pytest.mark.parametrize("route", ["plane", "plane_scalar", "line", "planar"])
def test_a_device_s_fluence_is_the_reference_s_at_every_node(monkeypatch, route):
    _require_device()
    _check_fluence(monkeypatch, route, "envelope")


@native_only
@pytest.mark.parametrize("n_pol", [1, 2])
def test_a_plane_s_widths_bring_only_numbers_across(n_pol, monkeypatch):
    """The session's plane statistics cross as one packed row, and no field does.

    The native session reduces the widths in its own pass
    (``session_statistics.cpp``); what crosses is the packed evaluation its plan
    unpacks, counted here. The session counts every field it exports, and the native
    state every whole-state transfer to the host by any path
    (``crossed_bytes()[0]``, solprop SEP 44): both stay at zero over the run.
    """
    statistics = pytest.importorskip("solprop_native._statistics")
    grid, problem, run_request = _simulation(
        "envelope", FreeGrid(3e-4, 32, 4e-4, 24), n_pol
    )
    request = run_request.for_run(stats=StatsSpec(names=WIDTHS))
    crossed = []
    unpack = statistics.StatisticsLowering.unpack

    def counted(self, packed):
        crossed.append(int(np.size(packed)))
        return unpack(self, packed)

    monkeypatch.setattr(statistics.StatisticsLowering, "unpack", counted)
    backend = select("native", problem, request, grid=grid)
    try:
        with launched(backend, problem, request) as (session, observation):
            native = session.native  # pyright: ignore[reportAttributeAccessIssue]
            assert native.status().exports == 0
            state = backend._native_state  # pyright: ignore[reportAttributeAccessIssue]
            assert state.crossed_bytes()[0] == 0
    finally:
        backend.close()
    assert set(observation.stats) == set(WIDTHS)
    # The two widths' numbers, against one per node.
    assert crossed
    assert 0 < max(crossed) < problem.transform.n_nodes / 4


@native_only
@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_widths_beside_the_other_statistics_read_the_same_volumes(kind):
    """One call forms the fluence and the peak from one pair of transverse volumes."""
    grid, problem, run_request = _route("plane", kind)
    field = _field_for("plane", grid, problem)
    together = (
        "energy",
        "on_axis_spectrum",
        "on_axis_intensity",
        "peak_intensity",
        *WIDTHS,
    )
    joint = _measure(
        "native", grid, problem, run_request, field, StatsSpec(names=together)
    )
    reference = _measure(
        "numpy_ref", grid, problem, run_request, field, StatsSpec(names=together)
    )
    bound = _width_budget(grid, problem)
    for name in together:
        one = _measure(
            "native", grid, problem, run_request, field, StatsSpec(names=(name,))
        )
        # The same statistic of the same state, in two orders of the same kernels.
        assert_within(joint[name], one[name], bound, what=f"joint {name}")
        assert_within(joint[name], reference[name], bound, what=f"reference {name}")


@native_only
@pytest.mark.parametrize("names", [WIDTHS, ("peak_intensity", *WIDTHS)])
def test_an_empty_plane_has_no_widths(names):
    grid, problem, run_request = _route("plane")
    field = np.zeros_like(problem.initial.field0)
    for backend in ("numpy_ref", "native"):
        values = _measure(
            backend, grid, problem, run_request, field, StatsSpec(names=names)
        )
        for name in WIDTHS:
            assert np.all(np.isnan(values[name]))


@native_only
def test_a_negative_lobe_is_refused_on_both():
    """A fluence negative beyond its rounding has no width, on either backend.

    Two plane waves in one bin, ``E = A + B exp(i k x)`` and
    ``G = Y0 A + Yk B exp(i k x)``, have ``Re(E G*) = (Y0 A - Yk B)(A - B)`` where they
    oppose. On half-micrometre pixels the highest column's admittance ``Yk`` is 17 %
    below the axis's ``Y0``, which puts ``A = 0.9 B`` between ``B`` and ``Yk B / Y0``:
    a negative lobe of about 0.2 % of the peak, against a budget of about 1e-12 of it.
    """
    space = FreeGrid(6e-6, 24, 6e-6, 24)
    grid, problem, run_request = _simulation(
        "envelope", space, 1, SpatialMedium(1.45), waist=2e-6
    )
    columns = np.zeros(problem.shape[0], complex)
    columns[0] = 0.9
    columns[12] = 1.0
    spectrum = np.zeros((1, problem.shape[2]), complex)
    spectrum[0, 0] = 1.0
    field = np.einsum("r,pw->rpw", columns, spectrum) * 1e6
    spec = StatsSpec(names=WIDTHS)
    for backend in ("numpy_ref", "native"):
        with pytest.raises(BackendError, match="nonnegative fluence profile"):
            _measure(backend, grid, problem, run_request, field, spec)


@native_only
def test_the_native_trace_needs_a_held_plane_and_real_rings():
    grid, problem, run_request = _route("plane")
    backend = select("native", problem, run_request, grid=grid)
    no_rings = np.empty(0)
    factor = 0.5 * EPS0 * C
    try:
        state = _load(backend, problem.initial.field0)
        with pytest.raises(RuntimeError, match="follows the call that holds it"):
            state.plane_fluence_trace(
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0, no_rings, no_rings
            )
        with pytest.raises(ValueError, match="invalid spatial statistic flags"):
            state.spatial_statistics(64, factor, 0.0)
        np.ravel(state.spatial_statistics(32, factor, 0.0))
        with pytest.raises(ValueError, match="invalid fluence rings"):
            state.plane_fluence_trace(
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0, no_rings, no_rings
            )
        np.ravel(state.spatial_statistics(32, factor, 0.0))
        with pytest.raises(ValueError, match="one-dimensional"):
            state.plane_fluence_trace(
                0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1, np.ones((1, 8)), np.ones((1, 8))
            )
    finally:
        backend.close()


@native_only
def test_a_line_has_no_plane_moments():
    grid, problem, run_request = _route("line")
    backend = select("native", problem, run_request, grid=grid)
    try:
        state = _load(backend, problem.initial.field0)
        with pytest.raises(ValueError, match="need a Cartesian plane"):
            state.spatial_statistics(32, 0.5 * EPS0 * C, 0.0)
    finally:
        backend.close()


def _ring_trace(samples, x, y):
    """The trace a backend forms: the reference's arithmetic, on the scaled samples."""

    def trace(*arguments):
        center_x, center_y, dx, dy, extent, spacing, n_radii, cosines, sines = arguments
        offset_x = np.arange(x.size) * dx - center_x
        offset_y = np.arange(y.size) * dy - center_y
        second = float(
            np.sum(
                samples
                * (
                    (offset_x[None, :] / extent) ** 2
                    + (offset_y[:, None] / extent) ** 2
                )
            )
        )
        averaged = np.empty(n_radii)
        for ring in range(n_radii):
            horizontal = np.clip(
                (center_x + spacing * ring * cosines) / dx, 0, x.size - 1
            )
            vertical = np.clip((center_y + spacing * ring * sines) / dy, 0, y.size - 1)
            ix = np.minimum(horizontal.astype(np.intp), x.size - 2)
            iy = np.minimum(vertical.astype(np.intp), y.size - 2)
            fx, fy = horizontal - ix, vertical - iy
            averaged[ring] = np.mean(
                (1 - fy) * ((1 - fx) * samples[iy, ix] + fx * samples[iy, ix + 1])
                + fy * ((1 - fx) * samples[iy + 1, ix] + fx * samples[iy + 1, ix + 1])
            )
        return np.array([[second, *averaged]])

    return trace


def _moments(profile, size=None, worst=None):
    maximum = float(np.max(profile))
    samples = profile / maximum
    n_y, n_x = profile.shape
    return np.array(
        [
            float(np.min(profile)) if worst is None else worst,
            0.0,
            maximum if size is None else size,
            maximum,
            0.0,
            float(np.sum(samples)),
            float(np.sum(samples * np.arange(n_x)[None, :])),
            float(np.sum(samples * np.arange(n_y)[:, None])),
        ]
    ), samples


def _plane_geometry():
    _, problem, _ = _route("plane")
    geometry = problem.transform.geometry
    assert geometry is not None
    return geometry


def test_plane_widths_read_the_reductions_as_the_host_widths_read_the_profile():
    geometry = _plane_geometry()
    x, y = geometry.axes
    grid_x, grid_y = np.meshgrid(x, y)
    profile = _gaussian(grid_x, grid_y, 3e-5, -2e-5)
    moments, samples = _moments(profile)
    got = plane_fluence_widths(geometry, moments, 100, _ring_trace(samples, x, y))
    want = cartesian_fluence_widths(x, y, profile.ravel())
    # One rule on one profile: only the summation order and the axes' accumulated
    # rounding differ, a few hundred units over the plane's nodes.
    assert_within(got, want, gamma(16 * profile.size))


def test_plane_widths_refuse_a_negative_lobe_beyond_its_budget():
    geometry = _plane_geometry()
    x, y = geometry.axes
    grid_x, grid_y = np.meshgrid(x, y)
    moments, samples = _moments(_gaussian(grid_x, grid_y), size=1.0, worst=-1e-6)
    with pytest.raises(BackendError, match="nonnegative fluence profile"):
        plane_fluence_widths(geometry, moments, 100, _ring_trace(samples, x, y))
    # Far below the peak, and within its budget: rounding, read as zero.
    within = moments.copy()
    within[0] = -0.5 * fluence_budget(np.array([1.0]), 100)[0]
    widths = plane_fluence_widths(geometry, within, 100, _ring_trace(samples, x, y))
    assert np.isfinite(widths[0])


def test_plane_widths_refuse_a_value_that_is_not_finite():
    geometry = _plane_geometry()
    moments = np.zeros(8)
    moments[4] = 1.0
    with pytest.raises(GridError, match="finite"):
        plane_fluence_widths(geometry, moments, 100, _unreachable)


def test_plane_widths_of_an_empty_plane_are_undefined():
    assert all(
        math.isnan(width)
        for width in plane_fluence_widths(
            _plane_geometry(), np.zeros(8), 100, _unreachable
        )
    )


def _unreachable(*arguments):
    raise AssertionError("a plane with nothing to trace was traced")


def test_plane_widths_refuse_a_second_moment_past_the_floats():
    geometry = _plane_geometry()
    x, y = geometry.axes
    grid_x, grid_y = np.meshgrid(x, y)
    moments, _ = _moments(_gaussian(grid_x, grid_y))

    def trace(*arguments):
        return np.array([math.inf, *np.ones(arguments[6])])

    with pytest.raises(GridError, match="not representable"):
        plane_fluence_widths(geometry, moments, 100, trace)


def test_plane_widths_at_an_edge_have_only_a_moment():
    geometry = _plane_geometry()
    x, y = geometry.axes
    grid_x, grid_y = np.meshgrid(x, y)
    profile = _gaussian(grid_x, grid_y, x[0], y[0], 1e-5)
    moments, samples = _moments(profile)
    got = plane_fluence_widths(geometry, moments, 100, _ring_trace(samples, x, y))
    want = cartesian_fluence_widths(x, y, profile.ravel())
    assert math.isnan(got[1]) and math.isnan(got[2])
    assert math.isnan(want[1]) and math.isnan(want[2])
    assert_within(got[0], want[0], gamma(16 * profile.size))


def test_the_rings_are_the_host_widths_rule():
    x = np.linspace(-1.0, 1.0, 9)
    y = np.linspace(-2.0, 2.0, 17)
    rings = _fluence_rings(x, y, 0.0, 0.0)
    assert rings is not None
    # Half the finest spacing, to the largest circle inside the axes about the centroid.
    assert rings.radii[1] == 0.125
    assert rings.radii[-1] == 1.0
    assert rings.cosines.size == rings.sines.size >= 8
    # A centroid within two half-spacings of an edge has none.
    assert _fluence_rings(x, y, 0.9, 0.0) is None
    # A Gaussian trace: its 1/e^2 radius and its FWHM, to the interpolant's accuracy.
    radii = np.linspace(0.0, 4.0, 4001)
    radius, width = _ring_fluence_widths(radii, np.exp(-2 * radii**2))
    assert radius == pytest.approx(1.0, abs=1e-6)
    assert width == pytest.approx(np.sqrt(2 * np.log(2)), abs=1e-6)


# --- the native finishing, against solprop's own (SEP 47 section 6.4) -------------


def _profiles(rng):
    """Radial or line profiles a width is most exposed on, with their axis values."""
    radius = np.sort(rng.uniform(1e-6, 4e-4, 300))
    gaussian = np.exp(-2 * (radius / 1e-4) ** 2)
    return radius, (
        (gaussian, 1.0),
        # A ring: the axis below every level, so no threshold width.
        (np.exp(-(((radius - 2e-4) / 3e-5) ** 2)), 0.0),
        # A flat top a ripple above its separately formed axis value.
        (np.where(radius < 1e-4, 1.0 + 1e-9, 0.0), 1.0),
        # A pedestal that never falls below the 1/e^2 level.
        (0.5 + 0.5 * gaussian, 1.0),
        # Nothing, and one sample.
        (np.zeros_like(radius), 0.0),
        (np.where(np.arange(radius.size) == 7, 1.0, 0.0), 0.0),
        (rng.uniform(0.0, 1.0, radius.size) * gaussian, 1.0),
    )


@native_only
def test_the_native_radial_widths_are_solprop_s_to_the_bit():
    """The radial widths ported operation for operation: NumPy's pairwise sums, the
    crossings' interpolation, the hollow and the uncrossed profiles."""
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _core,
    )

    from solprop.spatial import radial_fluence_widths  # noqa: PLC0415

    rng = np.random.default_rng(47)
    radius, cases = _profiles(rng)
    weights = rng.uniform(1e-10, 2e-9, radius.size)
    for size in (2, 7, 8, 9, 128, 129, 300):
        for fluence, axis in cases:
            arguments = (radius[:size], fluence[:size], weights[:size], axis)
            want = radial_fluence_widths(*arguments)
            got = _core.radial_fluence_widths(*arguments, math.exp(-2))
            np.testing.assert_array_equal(got, want, err_msg=f"{size} samples")


@native_only
def test_the_native_radial_widths_refuse_what_solprop_s_refuse():
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _core,
    )

    from solprop.spatial import radial_fluence_widths  # noqa: PLC0415

    radius = np.array([1e-5, 2e-5, 3e-5])
    weights = np.ones(3)
    for fluence, axis in (
        (np.array([1.0, -1e-300, 0.0]), 1.0),
        (np.array([1.0, np.nan, 0.0]), 1.0),
        (np.ones(3), -1.0),
        (np.ones(3), math.inf),
    ):
        with pytest.raises(GridError, match="radial fluence widths require") as want:
            radial_fluence_widths(radius, fluence, weights, axis)
        with pytest.raises(ValueError, match="radial fluence widths require") as got:
            _core.radial_fluence_widths(radius, fluence, weights, axis, math.exp(-2))
        assert str(got.value) == str(want.value)
    huge = np.full(3, 1e300)
    with pytest.raises(GridError, match="not representable"):
        radial_fluence_widths(radius * 1e200, huge, weights, 1e300)
    with pytest.raises(ValueError, match="not representable"):
        _core.radial_fluence_widths(radius * 1e200, huge, weights, 1e300, math.exp(-2))


X86_64 = platform.machine().lower() in {"x86_64", "amd64"}
"""Whether NumPy's compiled loops run unfused here: x86-64's distribution baselines have
no fused multiply-add, where an AArch64 build (Apple silicon, Graviton) may contract
``numpy.interp``'s product and sum."""


def unfused_interp(x, xp, fp):
    """``numpy.interp``'s compiled rule, each product and sum its own ufunc.

    ``numpy/_core/src/multiarray/compiled_base.c``: the end values outside the axis,
    the sample itself on a node and at the last node, else
    ``slope * (x - xp[j]) + fp[j]`` with ``slope = (fp[j+1] - fp[j]) / (xp[j+1] -
    xp[j])``. No platform fuses separate ufuncs, so this is the rule the native
    finishing ports, on every platform; on x86-64 it is ``numpy.interp`` to the bit.
    """
    x = np.asarray(x, dtype=np.float64)
    j = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, xp.size - 2)
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    offset = x - xp[j]
    product = slope * offset
    value = product + fp[j]
    value = np.where(x == xp[j], fp[j], value)
    value = np.where(x >= xp[-1], fp[-1], value)
    return np.where(x < xp[0], fp[0], value)


def host_interp(monkeypatch):
    """Hold the host finishing to the unfused rule where NumPy's own may fuse."""
    if not X86_64:
        monkeypatch.setattr(np, "interp", unfused_interp)


@pytest.mark.skipif(not X86_64, reason="NumPy's interp may fuse off x86-64")
def test_the_unfused_interp_is_numpy_s_on_x86_64():
    rng = np.random.default_rng(49)
    xp = np.linspace(-2e-4, 3e-4, 97)
    fp = rng.uniform(0.0, 1.0, xp.size)
    points = np.concatenate((rng.uniform(-3e-4, 4e-4, 1000), xp, xp[:1] - 1, xp[-1:]))
    np.testing.assert_array_equal(
        unfused_interp(points, xp, fp), np.interp(points, xp, fp)
    )


@native_only
def test_the_native_line_widths_are_solprop_s_to_the_bit(monkeypatch):
    """A line's centroid, moment and two-sided average, ``numpy.interp`` included: to
    the bit on x86-64, and to the unfused rule wherever NumPy's own may fuse."""
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _core,
    )

    from solprop.spatial import _line_fluence_widths  # noqa: PLC0415

    host_interp(monkeypatch)

    rng = np.random.default_rng(48)
    for n in (5, 8, 48, 129, 300):
        x = np.linspace(-2e-4, 3e-4, n)
        # Centred, near an edge, wider than the axis, and a point on one sample.
        point = float(x[n // 2])
        for centre, waist in (
            (0.0, 5e-5),
            (2.9e-4, 2e-5),
            (-1e-4, 3e-4),
            (point, 1e-6),
        ):
            samples = np.exp(-2 * ((x - centre) / waist) ** 2)
            samples = samples / np.max(samples)
            want = _line_fluence_widths(x, samples)
            got = _core.line_fluence_widths(x, samples, math.exp(-2))
            np.testing.assert_array_equal(got, want, err_msg=f"{n} {centre} {waist}")
        noisy = rng.uniform(0.0, 1.0, n)
        noisy = noisy / np.max(noisy)
        np.testing.assert_array_equal(
            _core.line_fluence_widths(x, noisy, math.exp(-2)),
            _line_fluence_widths(x, noisy),
        )


def _radial_route(kind="envelope"):
    from solprop.grid import RadialGrid  # noqa: PLC0415

    grid = _time_grid(kind)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[1e-3],
        dz_max=2.5e-4,
        dz_init=2.5e-4,
        absorber=None,
        integrator=RungeKutta(),
        stats=StatsSpec(names=WIDTHS),
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 6e-5
    )
    simulation = Simulation.from_radial(
        grid, RadialGrid(3e-4, 32), LOSSY, [], pulse, settings=settings, noise=None
    )
    return grid, simulation.translate(), simulation._run_request()


@native_only
@pytest.mark.parametrize("route", ["radial", "line", "planar"])
def test_a_profile_s_native_widths_refuse_as_the_host_s_do(monkeypatch, route):
    """The fluence block's finishing (``of_block``), on given blocks: the host's widths,
    its ``BackendError`` for a lobe beyond the rounding budget and its ``GridError``
    for a value that is not finite, each through ``translated`` as the plan raises
    it."""
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _core,
        _statistics,
    )

    host_interp(monkeypatch)
    grid, problem, run_request = _radial_route() if route == "radial" else _route(route)
    geometry = problem.transform.geometry
    assert geometry is not None
    lowering = _statistics.lower(
        problem,
        StatsSpec(names=WIDTHS),
        WIDTHS,
        grid=grid,
        z_end=float(run_request.stepper.z_end),
    )
    plan = lowering.plan
    assert plan is not None
    n_nodes = problem.transform.n_nodes
    n_terms = problem.shape[1] * grid.n_t + problem.transform.n_modes
    rng = np.random.default_rng(50)
    fluence = np.exp(-((np.arange(n_nodes) / (0.3 * n_nodes)) ** 2))
    fluence = fluence * rng.uniform(0.9, 1.0, n_nodes)
    axis = 1.0 if route == "radial" else math.nan
    largest = 1.0
    budget = fluence_budget(np.full(n_nodes, largest), n_terms)

    def both(profile):
        block = np.concatenate((profile, [axis, largest]))
        try:
            want = host_spatial.signed_fluence_widths(geometry, profile, budget, axis)
        except (BackendError, GridError) as error:
            with pytest.raises(ValueError) as caught:
                _core.fluence_block_widths(plan, block)
            raised = _statistics.translated(caught.value)
            assert type(raised) is type(error)
            assert str(raised) == str(error)
            return type(error)
        np.testing.assert_array_equal(_core.fluence_block_widths(plan, block), want)
        return None

    assert both(fluence) is None
    lobe = fluence.copy()
    lobe[3] = -1e-3
    assert both(lobe) is BackendError
    within = fluence.copy()
    within[3] = -0.5 * budget[3]
    assert both(within) is None
    broken = fluence.copy()
    broken[3] = math.nan
    assert both(broken) is GridError


@native_only
def test_a_state_that_is_not_finite_has_the_host_s_grid_error():
    """A radial state whose fluence is not finite: the native session's launch plane
    raises the reference's ``GridError``, through the plan and ``translated``."""
    grid, problem, run_request = _radial_route()
    field = np.array(problem.initial.field0, copy=True)
    # A bin the launch fills, so the active band keeps it.
    field[3, 0, int(np.argmax(np.abs(field[3, 0])))] = complex(math.nan, 0.0)
    spec = StatsSpec(names=WIDTHS)
    with pytest.raises(GridError, match="radial fluence widths require") as want:
        _measure("numpy_ref", grid, problem, run_request, field, spec)
    with pytest.raises(GridError, match="radial fluence widths require") as got:
        _measure("native", grid, problem, run_request, field, spec)
    assert str(got.value) == str(want.value)
    # And the session's plane, whose failure is the reference's exception too.
    from solprop_native.session import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeSession,
    )

    session = NativeSession(
        problem,
        run_request.for_run(stats=spec),
        Start(field=field, z=float(problem.initial.z)),
        grid=grid,
    )
    errors = []
    try:
        while not session.finished:
            batch = session.advance()
            errors += [
                event.error
                for event in batch.events
                if isinstance(event, Terminal) and event.error is not None
            ]
            session.acknowledge(batch.receipt)
    finally:
        session.close()
    assert [type(error) for error in errors] == [GridError]
    assert str(errors[0]) == str(want.value)

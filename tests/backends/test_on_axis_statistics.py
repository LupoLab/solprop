# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The on-axis plasma statistics of SEP 19 (section 5.2), on the reference backend.

They sit beside the unchanged node reductions: ``electron_density`` is the densest
node's, which a few-node field-scale rule may place well off the axis, while the
``on_axis_*`` names scan the field synthesised on the axis itself, with exactly the
tables and rule of the node scan. The checks: the axis value is the same whatever the
rule; a Radau rule's axis node supplies it without a further scan; a five-node Gauss
rule's node maximum misses the axial peak; a multimode field's off-axis maximum stays in
the legacy statistic; and every route and backend that does not define them refuses
them by name.
"""

import math
import warnings
from contextlib import closing
from typing import cast

import numpy as np
import pytest

from bounds import assert_within, device_backend, fft_bound, gamma, rate_log_sensitivity
from rate_oracles import adk_oracle
from solprop.backends import select
from solprop.backends.driver import StepDriver
from solprop.backends.host.driver import HostDriver
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.populations import axis_node
from solprop.driver import Simulation
from solprop.driver.simulation import _warn_sampled_peak
from solprop.exceptions import (
    BackendError,
    DriverError,
    MaterialRangeWarning,
    OutputError,
    SampledPeakWarning,
)
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.modes.quadrature import RadialRule, mode_matrix, on_axis_rule
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Gas
from solprop.pulses import GaussianPulse
from solprop.stats import ON_AXIS_PLASMA_STATISTICS, StatsSpec
from solprop.stepper import PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble
from solprop.translate.tdse_source import TDSE_UNAVAILABLE_STATISTICS

ENERGY = 1e-4  # J: a 10 fs pulse that ionises argon appreciably on the axis
AXIS = StatsSpec(names=ON_AXIS_PLASMA_STATISTICS)


@pytest.fixture(scope="module")
def grid():
    return FieldGrid.from_limits(0.02, 800e-9, (300e-9, 3e-6), 100e-15)


@pytest.fixture(scope="module")
def table():
    return build_rate_table(adk_oracle(Gas.Ar), (1e9, 2e11))


def argon(m=1, phi=0.0):
    return MarcatiliMode(125e-6, "Ar", 1.0, m=m, cladding=1.45, loss=False, phi=phi)


def problem_on(grid, table, modes, rule, start=None):
    """An argon capillary with Kerr and plasma on a given rule; nothing propagated."""
    if start is None:
        built = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=ENERGY)
        count = len(modes) if isinstance(modes, list) else 1
        start = np.repeat(built.build(grid), count, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return assemble(
            grid,
            modes,
            [KerrResponse(chi3=2.2e-52, thg=True), PlasmaResponse(Gas.Ar, table)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-16),
            save_planes=[0.02],
            dz_max=0.002,
            dz_init=1e-4,
            quadrature=rule,
        )


def entrance_stats(grid, problem, modes, spec):
    """The statistics of the launch field, and the backend's on-axis scan count."""
    with closing(NumpyRefBackend(problem, grid=grid, modes=modes)) as backend:
        backend.set_initial(problem.initial.field0, None)
        values = backend.stats(spec)
        # select() returns the seam's protocol; the counter is the shared driver's.
        return values, cast("HostDriver", backend).axis_scans


RULES: list[tuple[RadialRule, int]] = [
    ("gauss_kronrod", 32),
    ("field_scale", 5),
    ("field_scale", 8),
    ("field_scale_radau", 5),
]


def test_the_axis_value_is_the_same_on_every_rule(grid, table):
    """One axis field, one scan: the rule changes the nodes, not the axis.

    A Radau rule reads its axis node, synthesised by the transform rather than by the
    on-axis matrix, so the two agree to the rounding of the two syntheses.
    """
    mode = argon()
    values = []
    for radial_rule, n_r in RULES:
        rule = transverse_quadrature([mode], n_r=n_r, radial_rule=radial_rule)
        problem = problem_on(grid, table, [mode], rule)
        values.append(entrance_stats(grid, problem, [mode], AXIS)[0])
    reference = values[0]
    assert float(reference["on_axis_electron_density"]) > 0.0
    for other in values[1:]:
        for name in ON_AXIS_PLASMA_STATISTICS:
            # The same row, synthesised two ways, which rounding may separate by a
            # few ulp (measured equal here).
            np.testing.assert_allclose(other[name], reference[name], rtol=1e-12)


def test_the_default_rule_s_node_maximum_is_the_axis_value(grid, table):
    """Its innermost node is at eta = 1 - 1e-6: measured 2.1e-5 apart (about K 2e-6)."""
    mode = argon()
    rule = transverse_quadrature([mode], n_r=32, radial_rule="gauss_kronrod")
    problem = problem_on(grid, table, [mode], rule)
    spec = StatsSpec(names=("electron_density", "on_axis_electron_density"))
    values, _ = entrance_stats(grid, problem, [mode], spec)
    node = float(values["electron_density"])
    axis = float(values["on_axis_electron_density"])
    assert abs(node / axis - 1.0) < 1e-4


def test_five_gauss_nodes_miss_the_axial_peak_that_the_axis_statistic_reads(
    grid, table
):
    """The innermost of five Gauss field-scale nodes sees 92% of the axial intensity."""
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule="field_scale")
    problem = problem_on(grid, table, [mode], rule)
    spec = StatsSpec(names=("electron_density", "on_axis_electron_density"))
    values, scans = entrance_stats(grid, problem, [mode], spec)
    node = float(values["electron_density"])
    axis = float(values["on_axis_electron_density"])
    # Measured 0.44 of the axial value: a rate of order eight or so at 92%.
    assert node < 0.8 * axis
    assert scans == 1


@pytest.mark.parametrize(
    ("names", "radial_rule", "scans"),
    [
        (("species_ionisation", "on_axis_electron_density"), "field_scale_radau", 0),
        (("on_axis_electron_density",), "field_scale_radau", 1),
        (("species_ionisation", "on_axis_electron_density"), "field_scale", 1),
    ],
)
def test_a_radau_axis_node_supplies_the_row_when_the_nodes_were_scanned(
    grid, table, names, radial_rule, scans
):
    """Reuse within one observation only; otherwise at most one extra row."""
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule=radial_rule)
    problem = problem_on(grid, table, [mode], rule)
    _, counted = entrance_stats(grid, problem, [mode], StatsSpec(names=names))
    assert counted == scans


@pytest.mark.parametrize("radial_rule", ["field_scale", "field_scale_radau"])
def test_an_axis_statistic_leaves_the_node_statistics_the_plan_s(
    grid, table, radial_rule
):
    """SEP 47 section 6.4: one native implementation of each statistic, on a plane.

    Asking for an on-axis plasma statistic beside the node statistics moves only the
    axis statistic to the host: every other value is the native plan's, bit for bit
    the same with or without it, and a Radau rule's axis node still reuses the scan.
    """
    pytest.importorskip("solprop_native")
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule=radial_rule)
    problem = problem_on(grid, table, [mode], rule)
    nodes = (
        "energy",
        "peak_intensity",
        "electron_density",
        "peak_ionisation_rate",
        "species_ionisation",
        "species_electron_density",
        "species_ionisation_energy",
    )
    alone, _ = backend_stats("native", grid, problem, [mode], StatsSpec(names=nodes))
    spec = StatsSpec(names=(*nodes, *ON_AXIS_PLASMA_STATISTICS))
    both, counted = backend_stats("native", grid, problem, [mode], spec)
    assert counted == (0 if radial_rule == "field_scale_radau" else 1)
    assert set(both) == set(spec.names)
    for name in nodes:
        np.testing.assert_array_equal(both[name], alone[name], name)


@pytest.mark.parametrize("name", ["numpy_ref", "native"])
def test_consecutive_planes_scan_their_own_field(grid, table, name):
    """No row is reused across observations: each plane scans the field it has."""
    if name == "native":
        pytest.importorskip("solprop_native")
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule="field_scale")
    problem = problem_on(grid, table, [mode], rule)
    with closing(select(name, problem, grid=grid, modes=[mode])) as backend:
        backend.set_initial(problem.initial.field0, None)
        first = float(backend.stats(AXIS)["on_axis_electron_density"])
        backend.propagate_to(0.004)
        second = float(backend.stats(AXIS)["on_axis_electron_density"])
        # select() returns the seam's protocol; the counter is the driver's.
        assert cast("StepDriver | HostDriver", backend).axis_scans == 2
    assert first != second


def test_a_multimode_off_axis_maximum_stays_in_the_legacy_statistic(grid, table):
    """HE_11 minus HE_12, cancelled on the axis: the densest node is elsewhere."""
    modes = [argon(1), argon(2)]
    rule = transverse_quadrature(modes, n_r=32, radial_rule="gauss_kronrod")
    axis = mode_matrix(modes, on_axis_rule("polar"), components=("x",)).synthesis[0]
    built = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=ENERGY).build(grid)
    start = np.concatenate([built, -axis[0] / axis[1] * built], axis=0)
    problem = problem_on(grid, table, modes, rule, start=start)
    spec = StatsSpec(names=("electron_density", "on_axis_electron_density"))
    values, _ = entrance_stats(grid, problem, modes, spec)
    assert float(values["on_axis_electron_density"]) < 1e-6 * float(
        values["electron_density"]
    )


def test_the_species_forms_sum_to_the_total(grid, table):
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule="field_scale")
    problem = problem_on(grid, table, [mode], rule)
    values, _ = entrance_stats(grid, problem, [mode], AXIS)
    species = values["on_axis_species_electron_density"]
    assert species.shape == (problem.responses.n_species,)
    assert float(np.sum(species)) == float(values["on_axis_electron_density"])
    assert float(np.max(values["on_axis_species_peak_ionisation_rate"])) == float(
        values["on_axis_peak_ionisation_rate"]
    )


# --------------------------------------------------------------------- the refusals


@pytest.mark.parametrize("name", ["numpy_ref", "native"])
def test_the_mode_averaged_route_refuses_them_by_name(grid, table, name):
    problem = problem_on(grid, table, argon(), None)
    with pytest.raises(BackendError, match="the mode-averaged route is not qualified"):
        backend_stats(name, grid, problem, [argon()], AXIS)
    # An effective-area node is not the physical axis, even if its synthesis row
    # were supplied as the candidate: this route must never reuse it as the axis.
    candidate = np.repeat(problem.transform.synthesis[:1], 2, axis=0)
    assert axis_node(problem, candidate) is None


def test_the_driver_refuses_axis_statistics_before_saving_a_non_plasma_run():
    simulation = Simulation.from_capillary(
        125e-6,
        1e-6,
        "Ar",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(500e-9, 1.2e-6),
        trange=80e-15,
        fwhm=20e-15,
        energy=1e-7,
        plasma=False,
        stats=AXIS,
        backend="numpy_ref",
        noise=None,
    )
    with pytest.raises(DriverError, match="need a compiled plasma response"):
        simulation.run()


def test_a_problem_without_plasma_refuses_them(grid):
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule="field_scale")
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=ENERGY).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = assemble(
            grid,
            [mode],
            [KerrResponse(chi3=2.2e-52, thg=True)],
            start,
            controller=PIController(rtol=1e-8, atol=1e-16),
            save_planes=[0.02],
            dz_max=0.002,
            dz_init=1e-4,
            quadrature=rule,
        )
    with (
        closing(NumpyRefBackend(problem, grid=grid, modes=[mode])) as backend,
        pytest.raises(BackendError, match="need a compiled plasma response"),
    ):
        backend.set_initial(problem.initial.field0, None)
        backend.stats(AXIS)


BACKENDS = ["native"]


def backend_stats(name, grid, problem, modes, spec):
    """The statistics of the launch field on a named backend, and its scan count."""
    if name == "native":
        pytest.importorskip("solprop_native")
    with closing(select(name, problem, grid=grid, modes=modes)) as backend:
        backend.set_initial(problem.initial.field0, None)
        values = backend.stats(spec)
        # select() returns the seam's protocol; the counter is the driver's.
        return values, cast("StepDriver | HostDriver", backend).axis_scans


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize(
    ("radial_rule", "names", "scans"),
    [
        ("field_scale", ON_AXIS_PLASMA_STATISTICS, 1),
        ("field_scale_radau", ("species_ionisation", *ON_AXIS_PLASMA_STATISTICS), 0),
    ],
)
def test_the_compiled_backends_match_the_reference(
    grid, table, name, radial_rule, names, scans
):
    """SEP 19 phase 2: the same axis on every backend, and the same reuse.

    The axis row is one host scan per saved plane, shared by every backend, so a
    fresh scan agrees to rounding; a Radau rule's axis node comes from each backend's
    own node scan, whose parity with the reference is the node scan's.
    """
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule=radial_rule)
    problem = problem_on(grid, table, [mode], rule)
    spec = StatsSpec(names=names)
    want, _ = entrance_stats(grid, problem, [mode], spec)
    got, counted = backend_stats(name, grid, problem, [mode], spec)
    assert counted == scans
    for statistic in ON_AXIS_PLASMA_STATISTICS:
        # The node scans' parity tolerance: a compiled plasma column against the
        # reference's multipass scan over the window.
        np.testing.assert_allclose(got[statistic], want[statistic], rtol=1e-10)


def _axis_row(modes):
    rule = on_axis_rule(modes[0].dimlimits(0.0)[0])
    return mode_matrix(modes, rule, 0.0, components=("x", "y")).synthesis[0]


def _axis_state(grid, name):
    """Return one adversarial state's modes and spectrum, and how far the axis cancels.

    The native row sums the modes on the axis before the time transform and the
    reference after it, so that cancellation between modes amplifies both one's
    rounding: ``kappa`` is the largest term over the largest sum, on the row.
    """
    built = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=ENERGY).build(grid)
    if name == "one_mode":
        return [argon()], built, 1.0
    if name == "two_polarisations":
        return [argon()], np.concatenate([built, 0.3j * built], axis=1), 1.0
    if name == "rotated":
        # A mode with field on the axis's y row, which `phi = 0` leaves at zero, so that
        # a row dropping y would still read the same.
        modes = [argon(phi=math.pi / 4)]
        assert np.all(np.abs(_axis_row(modes)) > 0.0)
        return modes, built, 1.0
    if name == "one_bin":
        # All the energy in the peak bin: a tone of the pulse's peak amplitude.
        tone = np.zeros_like(built)
        peak = int(np.argmax(np.abs(built)))
        tone[..., peak] = np.sum(np.abs(built))
        return [argon()], tone, 1.0
    modes = [argon(1), argon(2)]
    profile = _axis_row(modes)
    if name == "two_modes":
        return modes, np.concatenate([built, 0.4j * built], axis=0), 1.0
    # The second mode cancels 90 % of the first on the axis.
    assert name == "cancelling"
    second = -0.9 * profile[0] / profile[1] * built
    return modes, np.concatenate([10 * built, 10 * second], axis=0), 19.0


AXIS_STATES = (
    "one_mode",
    "rotated",
    "two_polarisations",
    "one_bin",
    "two_modes",
    "cancelling",
)


def _check_axis_row(grid, table, name):
    modes, start, kappa = _axis_state(grid, name)
    rule = transverse_quadrature(modes, n_r=32, radial_rule="gauss_kronrod")
    problem = problem_on(grid, table, modes, rule, start=start)
    want, _ = entrance_stats(grid, problem, modes, AXIS)
    got, counted = backend_stats("native", grid, problem, modes, AXIS)
    assert counted == 1
    rate = float(np.max(want["on_axis_peak_ionisation_rate"]))
    assert rate > 0
    # The row's relative error, on both routes: the time transform and the modes' sum,
    # amplified by the axis's cancellation. A rate moves by its log sensitivity times
    # that, and a final fraction by no more: `1 - exp(-X)` has a log derivative of at
    # most one in X, and X's is a weighted mean of the rate's over the pulse. The rate
    # is steepest at its weakest, so the sensitivity is read an order of magnitude
    # below the peak's and doubled.
    sensitivity = rate_log_sensitivity(adk_oracle(Gas.Ar), 1e-1 * rate, (1e9, 2e11))
    row = kappa * (fft_bound(grid.n_to) + gamma(8 * len(modes)))
    bound = 2 * sensitivity * row
    for statistic in ON_AXIS_PLASMA_STATISTICS:
        assert_within(
            got[statistic], want[statistic], bound, what=f"{name} {statistic}"
        )


@pytest.mark.parametrize("name", AXIS_STATES)
def test_the_native_row_reads_the_reference_s_axis(grid, table, name):
    pytest.importorskip("solprop_native")
    _check_axis_row(grid, table, name)


@pytest.mark.gpu
@pytest.mark.parametrize("name", AXIS_STATES)
def test_a_device_s_row_reads_the_reference_s_axis(grid, table, name):
    # The row is formed in the device's memory space; only the row crosses.
    pytest.importorskip("solprop_native")
    if device_backend("native") == "none":
        pytest.skip("the installed solprop-native build has no accelerator backend")
    _check_axis_row(grid, table, name)


class _StateRefused:
    """A native state that refuses to bring its field across (solprop #359)."""

    def __init__(self, state):
        self._state = state

    def field_to_host(self):
        raise AssertionError("an on-axis statistic brought the state to the host")

    def __getattr__(self, name):
        return getattr(self._state, name)


@pytest.mark.parametrize("radial_rule", ["field_scale", "field_scale_radau"])
def test_the_native_axis_row_leaves_the_state_on_the_device(grid, table, radial_rule):
    pytest.importorskip("solprop_native")
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule=radial_rule)
    problem = problem_on(grid, table, [mode], rule)
    with closing(select("native", problem, grid=grid, modes=[mode])) as backend:
        backend.set_initial(problem.initial.field0, None)
        held = backend._native_state  # pyright: ignore[reportAttributeAccessIssue]
        backend._native_state = _StateRefused(held)  # pyright: ignore[reportAttributeAccessIssue]
        try:
            values = backend.stats(StatsSpec(names=AXIS.names))
        finally:
            backend._native_state = held  # pyright: ignore[reportAttributeAccessIssue]
    assert np.all(values["on_axis_peak_ionisation_rate"] > 0)


def test_the_native_row_refuses_a_matrix_that_is_not_one_or_two_rows(grid, table):
    pytest.importorskip("solprop_native")
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule="field_scale")
    problem = problem_on(grid, table, [mode], rule)
    with closing(select("native", problem, grid=grid, modes=[mode])) as backend:
        state = backend._native_state  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(RuntimeError, match="no initial field"):
            state.axis_spectrum(np.ones((1, 1)))
        backend.set_initial(problem.initial.field0, None)
        with pytest.raises(ValueError, match="at most two rows"):
            state.axis_spectrum(np.ones((3, 1)))
        with pytest.raises(ValueError, match="two-dimensional"):
            state.axis_spectrum(np.ones(1))


def test_a_host_backend_s_axis_scan_needs_the_field(grid, table):
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule="field_scale")
    problem = problem_on(grid, table, [mode], rule)
    with closing(NumpyRefBackend(problem, grid=grid, modes=[mode])) as backend:
        backend.set_initial(problem.initial.field0, None)
        with pytest.raises(BackendError, match="needs the field"):
            backend._axis_row_populations(  # pyright: ignore[reportPrivateUsage]
                None, np.zeros((2, 1))
            )


def test_they_are_saved_plane_statistics_and_tdse_never_defines_them():
    with pytest.raises(OutputError, match="between save planes"):
        StatsSpec(names=ON_AXIS_PLASMA_STATISTICS, periodic=ON_AXIS_PLASMA_STATISTICS)
    assert set(ON_AXIS_PLASMA_STATISTICS) <= TDSE_UNAVAILABLE_STATISTICS


# ---------------------------------------------------------------- the Q8 warning


def test_a_few_node_gauss_rule_warns_that_its_node_maximum_is_off_axis():
    mode = argon()
    spec = StatsSpec(names=("electron_density",))
    few = transverse_quadrature([mode], n_r=5, radial_rule="field_scale")
    with pytest.warns(SampledPeakWarning, match="eta = 0.958"):
        _warn_sampled_peak(few, spec)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SampledPeakWarning)
        # The Radau rule has a node on the axis, the default rule one at 1 - 1e-6,
        # and the on-axis statistics need no warning.
        radau = transverse_quadrature([mode], n_r=5, radial_rule="field_scale_radau")
        _warn_sampled_peak(radau, spec)
        default = transverse_quadrature([mode], n_r=32, radial_rule="gauss_kronrod")
        _warn_sampled_peak(default, spec)
        _warn_sampled_peak(few, AXIS)
        dense = transverse_quadrature([mode], n_r=64, radial_rule="field_scale")
        _warn_sampled_peak(dense, spec)


# ------------------------------------- envelope grid, mixture, taper, preionisation


def envelope_run(radial_rule, n_r):
    """Four fixed steps of an envelope Ar/He fill in a tapered bore, 10% preionised."""
    from solprop.driver import Simulation  # noqa: PLC0415 - the driver layer
    from solprop.physdata.mixtures import Mixture  # noqa: PLC0415
    from solprop.profiles import Linear  # noqa: PLC0415
    from solprop.stepper import FixedSchedule  # noqa: PLC0415

    names = ("electron_density", "species_electron_density", *ON_AXIS_PLASMA_STATISTICS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        warnings.simplefilter("ignore", SampledPeakWarning)
        simulation = Simulation.from_capillary(
            Linear(125e-6, 100e-6, 8e-6),
            8e-6,
            Mixture.fill({"Ar": 0.5, "He": 0.5}, 1.0),
            None,
            wavelength=800e-9,
            wavelength_limits=(500e-9, 1.2e-6),
            trange=80e-15,
            fwhm=20e-15,
            energy=3e-4,
            envelope=True,
            plasma={"Ar": adk_oracle("Ar")},
            preionisation={"Ar": 0.1},
            raman=False,
            kerr=False,
            n_r=n_r,
            radial_rule=radial_rule,
            save_planes=[4e-6, 8e-6],
            schedule=FixedSchedule.uniform(0.0, 8e-6, 2e-6),
            absorber_scheme="split_step",
            stats=StatsSpec(names=names),
            backend="numpy_ref",
            noise=None,
        )
        output = simulation.run()
    return {name: np.asarray(output[f"stats/{name}"]) for name in names}


def test_an_envelope_mixture_in_a_taper_reads_the_same_axis_on_every_rule():
    """Cycle-averaged rates, an Ar/He mixture, preionisation and a taper.

    At the entrance every rule sees the same field, so the axis statistics agree; the
    non-ionising helium stays at zero; the Radau rule's node maximum is its axis node;
    the five-node Gauss rule's is below the axis.
    """
    runs = {
        rule: envelope_run(rule, n_r)
        for rule, n_r in (
            ("gauss_kronrod", 32),
            ("field_scale", 5),
            ("field_scale_radau", 5),
        )
    }
    reference = runs["gauss_kronrod"]
    for values in runs.values():
        for name in ON_AXIS_PLASMA_STATISTICS:
            np.testing.assert_allclose(values[name][0], reference[name][0], rtol=1e-12)
        species = values["on_axis_species_electron_density"]
        assert np.all(species[:, 1] == 0.0)  # helium has no channel
        np.testing.assert_allclose(
            species.sum(axis=1), values["on_axis_electron_density"], rtol=1e-15
        )
    radau = runs["field_scale_radau"]
    np.testing.assert_allclose(
        radau["electron_density"][0], radau["on_axis_electron_density"][0], rtol=1e-12
    )
    gauss = runs["field_scale"]
    assert gauss["electron_density"][0] < gauss["on_axis_electron_density"][0]


def test_the_axis_reduction_checks_its_channel_data(grid, table):
    from solprop.backends.populations import axis_statistics  # noqa: PLC0415

    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule="field_scale")
    problem = problem_on(grid, table, [mode], rule)
    with pytest.raises(BackendError, match="channel arrays of shape"):
        axis_statistics(problem, 0.0, np.zeros(2), np.zeros(2))
    with pytest.raises(BackendError, match=r"fractions in \[0, 1\]"):
        axis_statistics(problem, 0.0, np.array([1.5]), np.array([0.0]))


def test_a_state_that_carries_its_components_reads_the_same_axis(grid, table):
    """Scalar transform rows and an (x, y) state: the profile's magnitude on the axis.

    The fresh axis scan of a Gauss rule and the Radau rule's own axis node, which the
    transform synthesises, agree.
    """
    mode = argon()
    built = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=ENERGY).build(grid)
    start = np.concatenate([built, 0.3 * built], axis=1)
    values = []
    for radial_rule in ("field_scale", "field_scale_radau"):
        rule = transverse_quadrature([mode], n_r=5, radial_rule=radial_rule)
        problem = problem_on(grid, table, [mode], rule, start=start)
        assert problem.shape[1] == 2
        assert problem.transform.n_pol == 1
        spec = StatsSpec(names=("species_ionisation", *ON_AXIS_PLASMA_STATISTICS))
        values.append(entrance_stats(grid, problem, [mode], spec))
    (gauss, fresh), (radau, reused) = values
    assert (fresh, reused) == (1, 0)
    np.testing.assert_allclose(
        gauss["on_axis_electron_density"], radau["on_axis_electron_density"], rtol=1e-12
    )


@pytest.mark.parametrize("name", ["numpy_ref", "native"])
def test_a_backend_without_its_modes_cannot_synthesise_the_axis(grid, table, name):
    if name == "native":
        pytest.importorskip("solprop_native")
    mode = argon()
    rule = transverse_quadrature([mode], n_r=5, radial_rule="field_scale")
    problem = problem_on(grid, table, [mode], rule)
    with (
        closing(select(name, problem, grid=grid)) as backend,
        pytest.raises(BackendError, match="need the mode set"),
    ):
        backend.set_initial(problem.initial.field0, None)
        backend.stats(AXIS)

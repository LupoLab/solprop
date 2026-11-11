# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify request-specific numerical observations without an execution engine."""

import gc
import json
import math
import subprocess
import sys
import textwrap
import weakref
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from backends.julia.cases import cpu_case, material_case
from mixture_cases import mixture_translation
from solprop.backends.julia import _observations
from solprop.backends.julia._descriptors import TableTransfer
from solprop.backends.julia._observations import lower_observations
from solprop.backends.populations import population_statistics
from solprop.boundaries import (
    SPLIT_RATE_BUCKETS,
    SplitCrossing,
    split_bucket_rates,
    split_crossing_step,
)
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.grid import FieldGrid
from solprop.modes import transverse_quadrature
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.monitor import FLOOR_MARGIN, BoundaryCheck, BoundaryMonitor
from solprop.output.schema import SavedPlane
from solprop.profiles import Constant, Linear
from solprop.pulses import GaussianPulse
from solprop.stats import CATALOGUE, StatsSpec, collect
from solprop.stepper import PIController
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.precision import SINGLE
from solprop.translate.problem import assemble
from solprop.translate.run_request import BoundaryRequest, MethodKind

NAMES = (
    "omega0",
    "energy",
    "energy_window",
    "peak_power",
    "peak_power_window",
    "peak_power_total",
    "fwhm_t",
    "fwhm_t_outer",
    "density",
    "pressure",
    "core_radius",
    "species_ionisation",
    "species_electron_density",
    "electron_density",
    "peak_ionisation_rate",
    "peak_intensity",
    "quadrature_error",
)


@pytest.fixture(scope="module")
def field_case():
    return material_case("field", "plasma")


@pytest.fixture(scope="module")
def envelope_case():
    return material_case("envelope", "plasma")


def _request(case, names=(), *, periodic=(), windows=(), period=1, monitor=None):
    spec = StatsSpec(
        names=tuple(names),
        periodic=tuple(periodic),
        energy_windows=np.asarray(windows, dtype=float).reshape(-1, 2),
        period=period,
    )
    return replace(
        case.request,
        observation=replace(case.request.observation, stats=spec),
        boundary=BoundaryRequest(monitor),
    )


def _arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif is_dataclass(value):
        for entry in fields(value):
            yield from _arrays(getattr(value, entry.name))
    elif isinstance(value, tuple):
        for entry in value:
            yield from _arrays(entry)


def _table(coefficients, span=(0.0, 0.04)):
    coefficients = np.asarray(coefficients, dtype=float)
    return PiecewiseChebyshev(
        coefficients,
        np.asarray(span),
        np.array([coefficients.shape[1] - 1], dtype=np.intp),
        np.empty(0),
    )


def test_full_catalogue_has_public_shapes_units_and_one_based_packing():
    _, grid, mode, _, case = mixture_translation()
    periodic = ("quadrature_error", "energy_window", "species_ionisation")
    request = _request(
        case,
        NAMES,
        periodic=periodic,
        period=3,
        windows=((-np.inf, grid.omega0), (grid.omega0, np.inf)),
    )
    prepared = lower_observations(
        case.problem,
        request,
        grid=grid,
        modes=(mode,),
        profiles={
            "density": Constant(3e24),
            "pressure": Constant(1.7),
            "core_radius": Constant(30e-6),
        },
    )
    expected = {
        "omega0": (1, 1),
        "energy": (1, 1),
        "energy_window": (2, 1, 1),
        "peak_power": (1,),
        "peak_power_window": (2, 1),
        "species_ionisation": (2, case.problem.transform.n_nodes),
        "species_electron_density": (2,),
    }
    offset = 1
    assert tuple(slot.name for slot in prepared.slots) == NAMES
    for slot in prepared.slots:
        assert slot.shape == expected.get(slot.name, ())
        assert slot.count == math.prod(slot.shape)
        assert slot.offset == offset
        assert slot.unit == CATALOGUE[slot.name].units
        offset += slot.count
    assert prepared.width == offset - 1
    assert prepared.periodic_indices == tuple(
        NAMES.index(name) + 1 for name in periodic
    )
    assert prepared.period == 3
    assert prepared.populations is not None
    assert prepared.populations.species_labels == case.problem.responses.labels
    assert prepared.populations.channel_species == (1, 2)
    assert prepared.populations.peak_species == 0
    assert prepared.intensity_factor == 0.5 * EPS0 * C
    for array in _arrays(prepared):
        assert array.flags.owndata and array.flags.f_contiguous
        assert not array.flags.writeable
    with pytest.raises(FrozenInstanceError):
        prepared.__setattr__("period", 4)


def test_multimode_slots_match_the_actual_numpy_catalogue_shapes():
    _, grid, mode, _, _ = mixture_translation()
    modes = (mode, replace(mode, m=2))
    rule = transverse_quadrature(modes, n_r=5, radial_rule="gauss_kronrod")
    pulse = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9).build(grid)
    field = np.concatenate((pulse, 0.7 * pulse), axis=0)
    case = assemble(
        grid,
        modes,
        [],
        field,
        quadrature=rule,
        components=("x",),
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[grid.length],
        dz_max=1e-4,
        dz_init=1e-5,
    )
    names = NAMES[:8]
    request = _request(
        case,
        names,
        periodic=("peak_power_window", "energy", "omega0"),
        windows=((-np.inf, grid.omega0), (grid.omega0, np.inf), (-np.inf, np.inf)),
    )
    actual = lower_observations(case.problem, request, grid=grid)
    oracle = collect(request.observation.stats, grid, SavedPlane(z=0.0, field=field))
    for slot in actual.slots:
        assert slot.shape == oracle[slot.name].shape
        assert slot.count == oracle[slot.name].size
    assert actual.slots[2].shape == (3, 2, 1)
    assert actual.slots[4].shape == (3, 2)
    assert actual.periodic_indices == (5, 2, 1)


@pytest.mark.parametrize("fixture", ["field_case", "envelope_case"])
def test_parseval_weights_match_independent_analytic_signal(fixture, request):
    case, grid, _ = request.getfixturevalue(fixture)
    selected = _request(case, ("energy",))
    prepared = lower_observations(case.problem, selected, grid=grid)
    rng = np.random.default_rng(441)
    samples = rng.normal(size=grid.n_t)
    if grid.kind == "field":
        spectrum = np.fft.rfft(samples)
        analytic_spectrum = np.zeros(grid.n_t, dtype=complex)
        analytic_spectrum[: len(spectrum)] = spectrum
        analytic_spectrum[1 : len(spectrum) - 1] *= 2
        analytic = np.fft.ifft(analytic_spectrum)
    else:
        samples = samples + 1j * rng.normal(size=grid.n_t)
        spectrum = np.fft.fft(samples)
        analytic = samples
    spectral_energy = np.sum(prepared.grid.parseval * np.abs(spectrum) ** 2)
    time_energy = grid.dt * np.sum(np.abs(analytic) ** 2)
    # Two FFTs plus linear reductions; the factor bounds their depth and additions.
    budget = 16 * grid.n_t * np.finfo(float).eps * time_energy
    assert budget > 0
    assert abs(spectral_energy - time_energy) <= budget
    assert np.array_equal(prepared.grid.time, grid.t)
    without_context = lower_observations(case.problem, selected)
    assert np.array_equal(
        without_context.grid.time,
        (np.arange(grid.n_t) - grid.n_t // 2) * grid.dt,
    )


def test_frequency_windows_are_half_open_and_keep_fft_order(envelope_case):
    case, grid, _ = envelope_case
    edge = grid.omega[3]
    prepared = lower_observations(
        case.problem,
        _request(case, ("energy_window",), windows=((-np.inf, edge), (edge, np.inf))),
    )
    assert np.array_equal(prepared.grid.omega, grid.omega)
    assert np.array_equal(prepared.grid.windows[:, 0], grid.omega < edge)
    assert np.array_equal(prepared.grid.windows[:, 1], grid.omega >= edge)
    assert not prepared.grid.windows[3, 0]
    assert prepared.grid.windows[3, 1]
    assert prepared.grid.windows.dtype == np.bool_


@pytest.mark.parametrize("name", ["fwhm_t", "fwhm_t_outer"])
def test_width_requires_original_matching_grid(field_case, name):
    case, grid, _ = field_case
    request = _request(case, (name,))
    with pytest.raises(BackendError, match=r"julia.*original time grid"):
        lower_observations(case.problem, request)
    lower_observations(case.problem, request, grid=grid)
    for wrong in (
        replace(grid, dt=2 * grid.dt),
        replace(grid, omega=grid.omega + 1e9),
        replace(grid, t=grid.t[:-1]),
    ):
        with pytest.raises(BackendError, match=r"julia.*does not match"):
            lower_observations(case.problem, request, grid=wrong)
    with pytest.raises(BackendError, match=r"julia.*finite and increasing"):
        lower_observations(case.problem, request, grid=replace(grid, t=grid.t[::-1]))


def test_unrequested_profiles_are_not_called_and_records_are_independent():
    case = cpu_case()

    class Unread(Constant):
        def __call__(self, z):
            raise AssertionError("unrequested profile was evaluated")

    first = lower_observations(
        case.problem,
        _request(case, ("pressure",)),
        profiles={"pressure": Linear(1.0, 3.0, 0.02), "density": Unread(7.0)},
    )
    second = lower_observations(
        case.problem,
        _request(case, ("density",)),
        profiles={"density": Constant(4e24)},
    )
    assert first.profiles.density is None and first.profiles.core_radius is None
    assert second.profiles.pressure is None
    assert second.profiles.density is not None
    assert second.profiles.density.value == 4e24
    record = first.profiles.record()
    assert tuple(record) == ("pressure",)
    error = record["pressure"]["relative_error"]
    assert isinstance(error, float) and error <= 1e-12
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    record["pressure"]["source"] = "mutated"
    assert first.profiles.record()["pressure"]["source"] != "mutated"
    assert first.slots[0].unit == "bar"


@pytest.mark.parametrize("representation", ["constant", "table", "expansion"])
def test_power_weight_representation_is_copied_without_refitting(representation):
    case = cpu_case()
    shape = case.problem.shape
    constant = np.linspace(1.0, 2.0, shape[-1])[None, None]
    table = _table(np.stack((constant, constant / 8))[None])
    weights = _table(np.array([[[1.0, 0.5], [0.0, 0.25]]]))
    terms = np.stack((constant, constant * 2))
    expansion = ParameterExpansion(
        ("witness",),
        np.array([[0.0, 1.0]]),
        np.array([1]),
        terms,
        weights,
    )
    problem = replace(
        case.problem,
        linear=replace(
            case.problem.linear,
            power_weight_representation=representation,
            power_weight_constant=constant if representation == "constant" else None,
            power_weight=table if representation == "table" else None,
            power_weight_expansion=expansion if representation == "expansion" else None,
        ),
    )
    actual = lower_observations(problem, _request(case, ("omega0",))).power_weight
    assert actual is not None
    if representation == "constant":
        assert actual.values is not None
        assert np.array_equal(actual.values[:, 0], constant.ravel())
        constant.setflags(write=True)
        constant[:] = 9
        assert actual.values[0, 0] == 1
    elif representation == "table":
        assert actual.table is not None
        assert np.array_equal(
            actual.table.coefficients,
            table.coefficients.reshape(1, 2, -1).transpose(2, 1, 0),
        )
        assert actual.weights is None
    else:
        assert actual.values is not None and actual.weights is not None
        assert np.array_equal(actual.values[:, 0, :], terms.reshape(2, -1).T)
        assert np.array_equal(
            actual.weights.coefficients, weights.coefficients.transpose(2, 1, 0)
        )
        assert actual.table is None
    for array in _arrays(actual):
        assert array.flags.owndata and not array.flags.writeable


def test_intensity_has_its_own_coarse_axis_and_the_source_taper():
    _, _, mode, _, _ = mixture_translation()
    grid = FieldGrid.from_limits(
        1e-3,
        800e-9,
        (400e-9, 2e-6),
        100e-15,
        ffac=6.0,
        fft_size="pow2",
    )
    initial = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9).build(grid)
    case = assemble(
        grid,
        mode,
        [],
        initial,
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[grid.length],
        dz_max=1e-4,
        dz_init=1e-5,
    )
    taper = _table([[1.2, 0.2]])
    transform = replace(case.problem.transform, taper_scale=taper)
    problem = replace(
        case.problem, route=replace(case.problem.route, transform=transform)
    )
    prepared = lower_observations(
        problem,
        _request(case, ("peak_intensity",)),
        modes=(mode,),
    )
    expected = mode_matrix(
        (mode,),
        on_axis_rule(mode.dimlimits(0.0)[0]),
        0.0,
        components=("x", "y"),
    ).synthesis
    assert prepared.axis_synthesis is not None
    assert np.array_equal(prepared.axis_synthesis, expected)
    assert grid.n_to / grid.n_t != 1
    assert not np.array_equal(
        prepared.axis_synthesis, expected * problem.grids.pad_scale
    )
    assert isinstance(prepared.taper, TableTransfer)
    assert np.array_equal(prepared.taper.coefficients[0, :, 0], [1.2, 0.2])
    assert not np.shares_memory(prepared.axis_synthesis, problem.transform.synthesis)


@pytest.mark.parametrize("fixture", ["field_case", "envelope_case"])
def test_population_legacy_selector_and_unrequested_absence(fixture, request):
    case, _, _ = request.getfixturevalue(fixture)
    prepared = lower_observations(
        case.problem,
        _request(case, ("electron_density", "peak_ionisation_rate")),
    )
    assert prepared.populations is not None
    assert prepared.populations.peak_species == 1
    assert prepared.populations.channel_species == (1,)
    assert prepared.populations.species_labels == ("medium",)
    assert prepared.populations.species_densities == (1e24,)
    assert prepared.populations.species_potentials.shape == (1,)
    assert not prepared.populations.species_potentials.flags.writeable
    assert prepared.populations.measure is None
    assert lower_observations(case.problem, _request(case)).populations is None


@pytest.mark.parametrize("fractions", [[[0.8, 0.2]], [[0.4, 0.4]]])
def test_sole_envelope_channel_keeps_the_actual_reference_peak_node(
    envelope_case, fractions
):
    case, _, _ = envelope_case
    original = case.problem.responses.envelope_plasma
    assert original is not None
    channel = replace(original.channels[0], label="active")
    plasma = replace(original, channels=(channel,))
    responses = replace(
        case.problem.responses,
        labels=("passive", "active"),
        species_densities=(_table([[3e24]]), _table([[1e24]])),
        envelope_plasma=plasma,
        spectral_responses=(plasma,),
    )
    # A valid two-node numerical map isolates the reduction policy from rate/FIR
    # accuracy: the input reductions are already fresh and finite.
    transform = replace(
        case.problem.transform,
        kind="modal",
        synthesis=np.ones((2, 1)),
        projection=np.full((1, 2), 0.5),
        projection_coarse=np.full((1, 2), 0.5),
        node_weights=np.ones(2),
    )
    problem = replace(
        case.problem,
        responses=responses,
        route=replace(case.problem.route, transform=transform),
    )
    fractions = np.asarray(fractions)
    peaks = np.array([[3.0, 9.0]])
    oracle = population_statistics(problem, 0.0, fractions, peaks)
    prepared = lower_observations(problem, _request(case, ("peak_ionisation_rate",)))
    assert prepared.populations is not None
    assert prepared.populations.channel_species == (2,)
    assert prepared.populations.peak_species == 2
    # Exact node selection, including a first-node tie; a mixture maximum would
    # report 9 instead of the actual single-population reference's 3.
    assert oracle["peak_ionisation_rate"] == 3.0
    assert peaks[0, np.argmax(fractions[0])] == oracle["peak_ionisation_rate"]
    assert np.max(peaks) != oracle["peak_ionisation_rate"]


def test_species_order_includes_nonionising_materials(field_case):
    case, _, _ = field_case
    channel = replace(case.problem.responses.plasma_channels[0], label="active")
    responses = replace(
        case.problem.responses,
        labels=("passive", "active"),
        species_densities=(_table([[3e24]]), _table([[1e24]])),
        plasma_channels=(channel,),
    )
    prepared = lower_observations(
        replace(case.problem, responses=responses),
        _request(case, ("species_ionisation", "peak_ionisation_rate")),
    )
    assert prepared.populations is not None
    assert prepared.populations.species_labels == ("passive", "active")
    assert prepared.populations.channel_species == (2,)
    assert prepared.populations.peak_species == 2
    assert prepared.populations.species_densities == (3e24, 1e24)
    assert prepared.slots[0].shape == (2, case.problem.transform.n_nodes)


def test_nonionising_species_have_a_mapping_but_total_plasma_needs_channels():
    case = cpu_case()
    actual = lower_observations(case.problem, _request(case, ("species_ionisation",)))
    assert actual.populations is not None
    assert actual.populations.channel_species == ()
    assert actual.populations.peak_species == 0
    for name in ("electron_density", "peak_ionisation_rate"):
        with pytest.raises(BackendError, match=r"julia.*need plasma"):
            lower_observations(case.problem, _request(case, (name,)))


@pytest.mark.parametrize("split", [False, True])
def test_temporal_policy_uses_shared_thresholds_and_authoritative_crossing(
    field_case, split
):
    case, grid, _ = field_case
    rate = np.linspace(0, 30.0, grid.n_t)
    crossing = SplitCrossing(
        (np.arange(grid.n_omega) % SPLIT_RATE_BUCKETS).astype(np.uint8)[None],
        1e-14,
        fastest=1e-10,
    )
    # Every shared bucket appears; the one-based conversion must preserve its order.
    problem = replace(
        case.problem,
        split_crossing=crossing,
        responses=replace(
            case.problem.responses,
            temporal_absorber=rate,
            temporal_absorber_fine=np.resize(rate, grid.n_to),
            absorber_scheme="split_step" if split else "nonlinear",
        ),
    )
    monitor = BoundaryMonitor.build(
        grid,
        rate,
        BoundaryCheck(budget=0.007, stop_factor=5.0, quiet_collar=True),
        columns=np.array([1.7]),
        split=split,
    )
    monitor.crossing = replace(crossing, width=9e-14)
    monitor._steps = 93
    monitor._launch = 42
    request = _request(case, monitor=monitor)
    actual = lower_observations(problem, request)
    temporal = actual.temporal
    assert temporal is not None
    expected = replace(monitor, crossing=crossing if split else None).watch_policy()
    for name in (
        "warn",
        "stop",
        "activity",
        "quiet_share",
        "quiet_checks",
        "quiet_collar",
        "cadence",
        "interval",
        "split",
    ):
        assert getattr(temporal.policy, name) == expected[name]
    assert temporal.policy.crossing_share == expected["share"]
    assert temporal.policy.launch_floor_ratio == 1e-12
    assert temporal.policy.floor_margin == FLOOR_MARGIN
    assert np.array_equal(temporal.regions, monitor.weights.T)
    assert np.array_equal(temporal.columns, [1.7])
    assert monitor._steps == 93 and monitor._launch == 42
    assert monitor.crossing.width == 9e-14
    assert actual.can_rest is not split
    if split:
        assert temporal.buckets is not None and actual.split_rate is not None
        limits = [
            split_crossing_step(float(value), crossing.width, crossing.factor)
            for value in split_bucket_rates(crossing.fastest)
        ]
        assert np.array_equal(temporal.policy.bucket_limits, limits)
        assert np.array_equal(temporal.buckets, crossing.buckets.T.astype(np.int64) + 1)
        assert temporal.buckets.dtype == np.int64
        assert np.array_equal(actual.split_rate, rate)
    else:
        assert actual.split_rate is None and temporal.buckets is None
        assert temporal.policy.bucket_limits.size == 0
    rate.setflags(write=True)
    rate[:] = 99
    assert temporal.regions[-1, 3] == 30


def test_split_rate_is_retained_without_a_monitor(field_case):
    case, grid, _ = field_case
    rate = np.linspace(0, 2, grid.n_t)
    problem = replace(
        case.problem,
        responses=replace(
            case.problem.responses,
            temporal_absorber=rate,
            temporal_absorber_fine=np.resize(rate, grid.n_to),
            absorber_scheme="split_step",
        ),
    )
    actual = lower_observations(problem, _request(case))
    assert actual.temporal is None and not actual.can_rest
    assert actual.split_rate is not None
    assert np.array_equal(actual.split_rate, rate)
    assert not np.shares_memory(actual.split_rate, rate)


def test_binary32_monitor_weights_are_widened_exactly_and_owned(field_case):
    case, grid, _ = field_case
    columns = np.array([1.7], dtype=np.float32)
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck(), columns=columns)
    monitor.weights = monitor.weights.astype(np.float32)
    regions = monitor.weights
    actual = lower_observations(case.problem, _request(case, monitor=monitor)).temporal
    assert actual is not None
    assert actual.columns.dtype == actual.regions.dtype == np.float64
    assert np.array_equal(actual.columns, columns.astype(np.float64))
    assert np.array_equal(actual.regions, regions.T.astype(np.float64))
    for prepared, original in ((actual.columns, columns), (actual.regions, regions)):
        assert prepared.flags.owndata and prepared.flags.f_contiguous
        assert not prepared.flags.writeable
        assert not np.shares_memory(prepared, original)
    columns[:] = 2.0
    regions[:] = 0.0
    assert actual.columns[0] == float(np.float32(1.7))
    assert np.any(actual.regions[:, :2] > 0.0)


def test_rest_eligibility_is_the_shared_route_method_intersection(field_case):
    case, grid, _ = field_case
    rate = np.ones(grid.n_t)
    problem = replace(
        case.problem,
        responses=replace(
            case.problem.responses,
            temporal_absorber=rate,
            temporal_absorber_fine=np.resize(rate, grid.n_to),
            absorber_scheme="nonlinear",
        ),
    )
    monitor = BoundaryMonitor.build(grid, rate, BoundaryCheck())
    request = _request(case, monitor=monitor)
    assert lower_observations(problem, request).can_rest
    low = replace(
        request, integration=replace(request.integration, kind=MethodKind.LOW_STORAGE)
    )
    assert not lower_observations(problem, low).can_rest
    assert not lower_observations(problem, _request(case)).can_rest
    no_rate = replace(
        problem,
        responses=replace(
            problem.responses, temporal_absorber=None, temporal_absorber_fine=None
        ),
    )
    assert not lower_observations(no_rate, request).can_rest


@pytest.mark.parametrize(
    "name",
    [
        "zdw",
        "on_axis_intensity",
        "beam_radius",
        "fwhm_r",
    ],
)
def test_unsupported_names_are_refused_before_fitting(name, monkeypatch):
    case = cpu_case()

    def forbidden(*args, **kwargs):
        raise AssertionError("profile fitting preceded name admission")

    monkeypatch.setattr(_observations, "lower_statistic_profile", forbidden)
    with pytest.raises(BackendError, match=f"julia.*{name}"):
        lower_observations(
            case.problem,
            _request(case, ("density", name)),
            profiles={"density": Constant(1.0)},
        )


@pytest.mark.parametrize("name", ["density", "pressure", "core_radius"])
def test_each_profile_requires_its_own_context(name):
    case = cpu_case()
    with pytest.raises(BackendError, match=f"julia {name}.*own"):
        lower_observations(case.problem, _request(case, (name,)), profiles={})


@pytest.mark.parametrize("name", ["energy_window", "peak_power_window"])
def test_window_outputs_require_at_least_one_window(name):
    case = cpu_case()
    with pytest.raises(BackendError, match=r"julia.*at least one energy window"):
        lower_observations(case.problem, _request(case, (name,)))


def test_empty_request_keeps_empty_slots_and_no_origin():
    case = cpu_case()
    request = _request(case)
    problem_ref, request_ref = weakref.ref(case.problem), weakref.ref(request)
    actual = lower_observations(case.problem, request)
    assert actual.width == 0 and actual.slots == () and actual.periodic_indices == ()
    assert actual.profiles.record() == {}
    assert actual.grid.windows.shape == (case.problem.shape[-1], 0)
    assert actual.axis_synthesis is None and actual.populations is None
    del case, request
    gc.collect()
    assert problem_ref() is None and request_ref() is None


def test_repeated_problem_preparation_does_not_cache_request_or_context(field_case):
    case, grid, _ = field_case
    first = lower_observations(case.problem, _request(case, ("energy",)))
    second = lower_observations(
        case.problem,
        _request(case, ("peak_power",), periodic=("peak_power",), period=5),
    )
    assert first.slots[0].name == "energy" and first.periodic_indices == ()
    assert second.slots[0].name == "peak_power" and second.periodic_indices == (1,)
    assert first.period == 1 and second.period == 5
    assert not np.shares_memory(first.grid.omega, second.grid.omega)
    assert not np.shares_memory(first.grid.omega, grid.omega)


def test_scalar_admission_precedes_observation_preparation():
    case = cpu_case()
    for problem, message in (
        (
            replace(
                case.problem,
                precision=SINGLE,
                initial=replace(
                    case.problem.initial,
                    noise_model="input",
                    noise0=np.zeros(case.problem.shape, dtype=np.complex128),
                ),
                flags=replace(case.problem.flags, has_noise=True),
            ),
            "binary32 input noise",
        ),
        (
            replace(case.problem, shape=(1, 2, case.problem.shape[-1])),
            "vector polarisation",
        ),
    ):
        with pytest.raises(BackendError, match=f"julia.*{message}"):
            lower_observations(problem, _request(case))
    with pytest.raises(BackendError, match=r"julia peak_intensity.*matching mode set"):
        lower_observations(case.problem, _request(case, ("peak_intensity",)))


def test_binary32_observation_lowering_preserves_owned_wide_inputs():
    case = cpu_case()
    problem = replace(case.problem, precision=SINGLE)
    transfer = lower_observations(problem, _request(case, ("energy",)))
    assert transfer.width == problem.shape[0]
    assert transfer.grid.parseval.dtype == np.float64
    assert transfer.power_weight is not None
    assert transfer.power_weight.values is not None
    assert transfer.power_weight.values.dtype == np.float64
    assert not transfer.power_weight.values.flags.writeable
    assert not np.shares_memory(transfer.grid.omega, problem.grids.omega)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.inf, np.nan, 1 + 1j])
def test_invalid_constant_power_weight_is_named(invalid):
    case = cpu_case()
    # Deliberately corrupt a private descriptor to exercise this boundary as well
    # as the upstream constructor's usual dtype guard.
    linear = replace(case.problem.linear)
    object.__setattr__(
        linear, "power_weight_constant", np.full(case.problem.shape, invalid)
    )
    problem = replace(case.problem, linear=linear)
    with pytest.raises(
        BackendError, match=r"julia.*power weights.*real positive finite"
    ):
        lower_observations(problem, _request(case))


@pytest.mark.parametrize("representation", ["table", "expansion"])
@pytest.mark.parametrize("invalid", [np.inf, np.nan, 1 + 1j])
def test_nonreal_or_nonfinite_weight_tables_are_named(representation, invalid):
    case = cpu_case()
    table = _table([[[1.0]]])
    table = replace(table, coefficients=np.full((1, 1, 1), invalid))
    expansion = ParameterExpansion(
        ("witness",),
        np.array([[0.0, 1.0]]),
        np.array([0]),
        np.ones((1, *case.problem.shape)),
        table,
    )
    linear = replace(
        case.problem.linear,
        power_weight_constant=None,
        power_weight_representation=representation,
        power_weight=table if representation == "table" else None,
        power_weight_expansion=expansion if representation == "expansion" else None,
    )
    with pytest.raises(
        BackendError, match="julia power-weight tables must be real finite"
    ):
        lower_observations(replace(case.problem, linear=linear), _request(case))


def test_split_monitor_refuses_rate_or_bucket_mismatch(field_case):
    case, grid, _ = field_case
    rate = np.ones(grid.n_t)
    problem = replace(
        case.problem,
        responses=replace(
            case.problem.responses,
            temporal_absorber=rate,
            temporal_absorber_fine=np.ones(grid.n_to),
            absorber_scheme="split_step",
        ),
    )
    monitor = BoundaryMonitor.build(grid, rate * 2, BoundaryCheck(), split=True)
    with pytest.raises(BackendError, match="julia split monitor rate must equal"):
        lower_observations(problem, _request(case, monitor=monitor))
    monitor = BoundaryMonitor.build(grid, rate, BoundaryCheck(), split=True)
    crossing = SplitCrossing(np.zeros((1, grid.n_omega - 1), dtype=np.uint8), 1e-14)
    with pytest.raises(BackendError, match="julia split crossing buckets"):
        lower_observations(
            replace(problem, split_crossing=crossing), _request(case, monitor=monitor)
        )
    crossing = SplitCrossing(
        np.zeros((1, grid.n_omega), dtype=np.uint8), 1e-14, share=1.0
    )
    with pytest.raises(BackendError, match="julia split crossing share"):
        lower_observations(
            replace(problem, split_crossing=crossing), _request(case, monitor=monitor)
        )


def test_observation_lowering_does_not_inherit_source_noise_admission(monkeypatch):
    from solprop.backends.julia import _descriptors  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        raise AssertionError("observation lowering entered source admission")

    monkeypatch.setattr(_descriptors, "lower_modal", forbidden)
    case = cpu_case()
    problem = replace(case.problem, flags=replace(case.problem.flags, has_noise=True))
    actual = lower_observations(problem, _request(case, ("energy",)))
    assert actual.slots[0].name == "energy"


@pytest.mark.parametrize("budget,factor", [(1e300, 1.0), (1e153, 1e308)])
def test_finite_threshold_inputs_cannot_publish_an_overflowed_policy(
    field_case, budget, factor
):
    case, grid, _ = field_case
    monitor = BoundaryMonitor.build(
        grid, None, BoundaryCheck(budget=budget, stop_factor=factor)
    )
    with pytest.raises(
        BackendError, match=r"julia temporal monitor.*(overflowed|finite)"
    ):
        lower_observations(case.problem, _request(case, monitor=monitor))


@pytest.mark.parametrize(
    "corruption", ["regions", "columns", "kind", "interval", "split"]
)
def test_invalid_monitor_is_refused_without_state_mutation(field_case, corruption):
    case, grid, _ = field_case
    monitor = BoundaryMonitor.build(grid, None, BoundaryCheck())
    if corruption == "regions":
        monitor.weights = np.full((4, grid.n_t), np.nan)
    elif corruption == "columns":
        monitor.columns = np.full(1, -1.0)
    elif corruption == "kind":
        monitor.kind = "envelope"
    elif corruption == "interval":
        monitor.interval = math.nan
    else:
        monitor.split = True
    monitor._steps = 91
    with pytest.raises(BackendError, match=r"julia.*monitor"):
        lower_observations(case.problem, _request(case, monitor=monitor))
    assert monitor._steps == 91


def test_fresh_process_lowers_without_native_or_julia_import(monkeypatch, tmp_path):
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        class RefuseEngines(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                engines = ('solprop_native', 'juliacall', 'juliapkg')
                if fullname.split('.')[0] in engines:
                    raise AssertionError('execution engine imported: ' + fullname)
        sys.meta_path.insert(0, RefuseEngines())
        from backends.julia.cases import cpu_case
        from solprop.backends.julia._observations import lower_observations
        case = cpu_case()
        transfer = lower_observations(case.problem, case.request)
        assert transfer.width == 0
    """)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr

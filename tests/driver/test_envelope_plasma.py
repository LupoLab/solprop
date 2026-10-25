# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise envelope plasma through constructors, material updates and saved runs."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import cast

import numpy as np
import pytest

from bounds import rate_log_sensitivity
from driver.test_resume import run_until
from march_bounds import (
    FIXED_ORDER,
    PRESCRIBED,
    STATISTIC_FACTOR,
    assert_stored_matched,
    simulation_bound,
)
from rate_oracles import AdkOracle, adk_oracle
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.driver.executors import Batch, Local
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.modes import MarcatiliMode
from solprop.nonlinear import EnvelopePlasmaResponse, responses_for
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.raman import RamanResponse
from solprop.nonlinear.responses import KerrResponse
from solprop.output import open_file
from solprop.output.hdf5 import resume
from solprop.physdata import Mixture
from solprop.processing import node_fluence, species_ionisation
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule

# The temporal FFT extends outside the argon index fit. The active source band
# remains unchanged; extrapolation is unrelated to these interface witnesses.
pytestmark = pytest.mark.filterwarnings(
    f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
)

NAMES = (
    "energy",
    "electron_density",
    "peak_ionisation_rate",
    "species_ionisation",
    "species_density",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_ionisation_energy",
)


RATE_STATISTICS = ("peak_ionisation_rate", "species_peak_ionisation_rate")
"""The statistics that are a steep function of the field, not a quadratic form of it."""


def peak_rate_sensitivity(expected) -> float:
    """Return the largest `d ln W / d ln F` of the argon rate over a saved rate series.

    The statistic is cycle-averaged (the run's metadata says so), and a weaker peak
    has the steeper logarithm, so the smallest positive entry sets it.
    """
    values = np.asarray(expected, dtype=float)
    smallest = float(np.min(values[values > 0.0]))
    return rate_log_sensitivity(
        adk_oracle("Ar", cycle_averaged=True), smallest, (1e7, 1e12)
    )


def build(*, backend="numpy_ref", waveguide=False, **overrides):
    if backend == "native":
        pytest.importorskip("solprop_native")
    options = {
        "lambda0": 800e-9,
        "wavelength_limits": (500e-9, 1.2e-6),
        "trange": 80e-15,
        "fwhm": 20e-15,
        "energy": 1e-5,
        "envelope": True,
        "plasma": adk_oracle("Ar"),
        "preionisation": 0.1,
        "raman": False,
        "kerr": False,
        "n_r": 4,
        "radial_rule": "gauss",
        "save_planes": [4e-6, 8e-6],
        "schedule": FixedSchedule.uniform(0.0, 8e-6, 2e-6),
        # No order choice on the prescribed schedule: two runs are one discrete march.
        "integrator": FIXED_ORDER,
        "stats": StatsSpec(
            names=NAMES, periodic=("electron_density", "peak_ionisation_rate")
        ),
        "backend": backend,
    } | overrides
    if waveguide:
        simulation = Simulation.from_waveguide(
            [MarcatiliMode(125e-6, "Ar", 1.0, loss=False, cladding=1.45)],
            8e-6,
            "Ar",
            1.0,
            **options,
        )
    else:
        simulation = Simulation.from_capillary(
            125e-6,
            8e-6,
            "Ar",
            1.0,
            loss=False,
            cladding=1.45,
            **options,
        )
    # Keep four fixed steps inside the explicit collar's stability limit.
    simulation.settings = replace(
        simulation.settings, absorber=absorber(simulation.grid, boundary_length=1e-3)
    )
    return simulation


@pytest.mark.parametrize("waveguide", [False, True])
@pytest.mark.parametrize("rate", [True, adk_oracle("Ar"), {"Ar": adk_oracle("Ar")}])
def test_gas_constructors_select_envelope_plasma(waveguide, rate):
    simulation = build(waveguide=waveguide, plasma=rate)
    response = next(
        one for one in simulation.responses if isinstance(one, EnvelopePlasmaResponse)
    )
    # plasma=True is optmat's rate since SEP 3's Phase 4a; a caller's rate (the
    # test-only ADK oracle) is taken as it is.
    assert isinstance(response.rate, OptmatRate if rate is True else AdkOracle)
    assert response.rate.convention == "instantaneous-isotropic"
    assert simulation._mixture_selection is not None
    assert simulation._mixture_selection.envelope


def test_envelope_mixture_builder_defaults_and_rate_conventions():
    mixture = Mixture.fill({"Ar": 0.5, "He": 0.5}, 1.0)
    responses = responses_for(mixture, envelope=True, plasma=True, lambda0=800e-9)
    assert sum(isinstance(one, EnvelopePlasmaResponse) for one in responses) == 2
    assert all(
        not one.thg
        for one in responses
        if isinstance(one, KerrResponse | RamanResponse)
    )
    with pytest.raises(MaterialError, match="THG"):
        responses_for(mixture, envelope=True, plasma=True, thg=True, lambda0=800e-9)
    with pytest.raises(MaterialError, match="instantaneous"):
        responses_for(
            mixture,
            envelope=True,
            plasma={"Ar": adk_oracle("Ar", cycle_averaged=True)},
            lambda0=800e-9,
        )


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_envelope_mixture_update_reuses_rates_and_matches_fresh(backend):
    source = build(backend=backend)
    original = source.translate()
    mode = source.modes[0]
    assert isinstance(mode, MarcatiliMode)
    mixture = mode.mixture
    assert mixture is not None
    changed = replace(
        mixture,
        components=tuple(
            replace(component, preionisation=0.25) for component in mixture.components
        ),
    ).with_fill({"Ar": 1.0}, 0.8)
    updated = source.with_mixture(changed)
    problem = updated.translate()
    assert isinstance(updated.responses[0], EnvelopePlasmaResponse)
    assert (
        problem.responses.population_channels[0].rate
        is original.responses.population_channels[0].rate
    )
    assert problem.transform.synthesis is original.transform.synthesis
    assert problem.transform.projection is original.transform.projection
    fresh = Simulation(
        source.grid,
        [replace(mode, mixture=changed)],
        responses_for(
            changed, envelope=True, plasma={"Ar": adk_oracle("Ar")}, kerr=False
        ),
        source.initial,
        settings=source.settings,
    )
    actual, expected = updated.run(progress="none"), fresh.run(progress="none")
    # Both translations use the same table and quadrature budgets. This bound
    # covers four seven-stage FFT/source reductions plus table reconstruction.
    bound = 128 * source.grid.n_to * 7 * 4 * np.finfo(float).eps
    np.testing.assert_allclose(
        actual.fields(),
        expected.fields(),
        rtol=bound,
        atol=bound * np.max(np.abs(expected.fields())),
    )
    for name in NAMES:
        np.testing.assert_allclose(
            actual[f"stats/{name}"], expected[f"stats/{name}"], rtol=2e-6, atol=1e-30
        )


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
@pytest.mark.parametrize("noise", [None, "input", "modified"])
def test_envelope_hdf5_resume_and_reconstruction(
    tmp_path, backend, noise, record_property
):
    options = {"backend": backend, "noise": noise, "seed": 821 if noise else None}
    whole = build(**options).run(progress="none")
    path = tmp_path / "envelope.h5"
    run_until(build(**options), path, after=1)
    continued = resume(path).march is not None
    build(**options).run(path, resume=True, progress="none")
    with open_file(path) as restored:
        scale = float(np.max(np.abs(whole.fields())))
        departure = float(np.max(np.abs(restored.fields() - whole.fields()))) / scale
        record_property("power_resume_field", departure)
        # Where the march's state is kept (the host engines) a resume continues the
        # Adams history (ADR-046), so the resumed run is the same march: a predeclared
        # 2.7e-16 global field-amplitude bound for the power conversion's two
        # roundings, plus the two runs' own rounding budget (SEP 25). An engine that
        # starts the history afresh (Kokkos) restarts at order one: measured 5.2e-7
        # on the field and 1.3e-6 on a peak rate, bounded at 1e-5.
        matched = 2.7e-16 + simulation_bound(build(**options))
        fresh = 0.0 if continued else 1e-5
        assert departure <= max(matched, fresh)
        comparisons = {
            name: (restored[f"stats/{name}"], whole[f"stats/{name}"]) for name in NAMES
        }
        comparisons["node_fluence"] = (node_fluence(restored), node_fluence(whole))
        comparisons["species_ionisation"] = (
            species_ionisation(restored),
            species_ionisation(whole),
        )
        for name, (actual, expected) in comparisons.items():
            scale = float(np.max(np.abs(expected)))
            shift = float(np.max(np.abs(actual - expected)))
            record_property(f"power_resume_{name}", shift / scale if scale else shift)
            # The two amplitude conversions seed
            # round-off in subsequent FFT/response reductions; 32 eps bounds
            # these small, weakly evolving fixtures, and the two runs' own rounding
            # adds the march's budget, a statistic quadratic in the field twice
            # it or so (STATISTIC_FACTOR). Measured maxima: energy 1.69e-16, node
            # fluence 1.53e-16; other rows zero. A peak rate is `W(F)` of the field,
            # so it moves by `S = d ln W / d ln F` times the field's relative shift:
            # at these weak fields the ADK oracle's `S` is about 83, and the rates
            # measured 23 eps on an Apple-silicon laptop and 61 on an x86-64 server,
            # whose last bits of the seed differ.
            budget = 32 * np.finfo(float).eps + STATISTIC_FACTOR * matched
            if name in RATE_STATISTICS:
                budget *= max(1.0, peak_rate_sensitivity(expected))
            budget = max(budget, fresh)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=budget * scale)
        assert restored.metadata is not None
        assert restored.metadata.schema_version == 25
        assert restored.metadata.configuration["statistic_rate_conventions"] == {
            "peak_ionisation_rate": "cycle-averaged",
            "species_peak_ionisation_rate": "cycle-averaged",
        }


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numba_cpu", "native"])
@pytest.mark.parametrize("envelope", [False, True])
def test_modified_noise_plasma_statistics_scan_the_committed_field(
    backend, envelope, record_property
):
    """Under modified noise the periodic population statistics are the committed
    field's, as numpy_ref defines them, not the noisy last stage input's.

    Reusing the accepted step's last stage, whose input carries the noise draw,
    moved the field run's peak rate by 2.26e-5 relative while the fields agreed to
    1.6e-16.
    """
    options = {
        "envelope": envelope,
        "noise": "modified",
        "seed": 821,
        "wavelength_limits": (400e-9, 2e-6),
    }
    expected = build(**options).run(progress="none")
    actual = build(backend=backend, **options).run(progress="none")
    eps = np.finfo(float).eps
    scale = float(np.max(np.abs(expected.fields())))
    # The backends agree on the field to rounding: 32 eps bounds the few steps'
    # transforms and response sums (measured 1.6e-16 on the field grid).
    np.testing.assert_allclose(
        actual.fields(), expected.fields(), rtol=0, atol=32 * eps * scale
    )
    for name in ("electron_density", "peak_ionisation_rate"):
        reference = np.asarray(expected[f"stats/{name}"])
        value = np.asarray(actual[f"stats/{name}"])
        scale = float(np.max(np.abs(reference)))
        assert scale > 0
        record_property(name, float(np.max(np.abs(value - reference))) / scale)
        # A statistic of the same committed field moves with the field's rounding;
        # a peak rate by `d ln W / d ln F` times it (the chain rule), with the
        # instantaneous rate's own sensitivity on the field grid.
        budget = 32 * eps
        if name == "peak_ionisation_rate":
            smallest = float(np.min(reference[reference > 0.0]))
            rate = adk_oracle("Ar", cycle_averaged=envelope)
            budget *= max(1.0, rate_log_sensitivity(rate, smallest, (1e7, 1e12)))
        assert budget < 2.26e-5 / 10
        np.testing.assert_allclose(value, reference, rtol=0, atol=budget * scale)


def scan_point(point, backend):
    return build(
        backend=backend,
        energy=1e-5 * point.values["scale"],
        noise="modified",
        seed=point.seed,
    )


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_envelope_scan_batches_match_singletons_and_partial_tail(tmp_path, backend):
    scan = Scan("envelope", base_seed=711, scale=[0.8, 1.0, 1.1])
    builder = partial(scan_point, backend=backend)
    options = {"fields": "planes", "field_planes": [-1], "progress": "none"}
    single = run_scan(
        scan, builder, output=tmp_path / "single.h5", executor=Local(), **options
    )
    batch = run_scan(
        scan, builder, output=tmp_path / "batch.h5", executor=Batch(2), **options
    )
    assert single.status.tolist() == batch.status.tolist() == ["done"] * 3
    for point in scan.points():
        actual, expected = batch.point(point.index), single.point(point.index)
        assert (
            cast(Mapping, actual["metadata"])["configuration"]
            == cast(Mapping, expected["metadata"])["configuration"]
        )
        arrays = {
            name: np.asarray(value)
            for name, value in expected.items()
            if name != "metadata"
        }
        # A batch member is its singleton's march, to the march's rounding budget; a
        # rate statistic moves by its sensitivity `d ln W / d ln F` times the field.
        factors = {
            name: STATISTIC_FACTOR * max(1.0, peak_rate_sensitivity(value))
            for name, value in arrays.items()
            if any(rate in name for rate in RATE_STATISTICS) and np.any(value)
        }
        assert_stored_matched(
            arrays,
            {name: value for name, value in actual.items() if name != "metadata"},
            simulation_bound(builder(point), backend=backend),
            exact=PRESCRIBED,
            factors=factors,
            what=f"batch member {point.index}",
        )

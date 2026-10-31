# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ten held-current steps under the band rule, against Richardson (SEP 18, item 5).

Section 7 item 5's protocol, complete: a single-mode 125 um capillary stepped ten fixed
steps on ``numpy_ref`` with the split-step absorber, helium at 0.8 bar over 5 mm with
150 uJ and argon at 0.4 bar over 2 mm with 30 uJ, each launched as a 1.3 fs, 800 nm
Gaussian with three 4 fs seeds of 2e-4 of its energy. The quantity is the
plasma-induced change of the complex coarse exit spectrum -- plasma on minus plasma off
-- and the reference is the Richardson value of the plain trapezoid at ``ffac`` 48 and
96, whose uncertainty is the same extrapolation from 24 and 48. With the plasma off
every ``ffac`` agrees to 1e-12, so the plasma-off run at ``ffac = 96`` is the baseline.
"""

import dataclasses
import warnings
from typing import Any, cast

import numpy as np
import pytest

from convergence_records import Column, Study, record
from driver.test_prop_capillary import build
from legacy_geometry import LEGACY_GRID
from solprop.driver import prop_capillary
from solprop.exceptions import (
    BackendError,
    DriverError,
    MaterialError,
    TranslationError,
)
from solprop.grid import FieldGrid
from solprop.nonlinear.mixtures import responses_for
from solprop.nonlinear.responses import PlasmaResponse
from solprop.output.schema import serialise
from solprop.profiles import Constant
from solprop.pulses import GaussianPulse
from solprop.stats import PLASMA_INTEGRATION_STATISTICS, StatsSpec
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.plasma_integration import design_running_integral

CASES = {
    # gas: length in m, wavelength limits in m, pressure in bar, energy in J, seeds
    "He": (5e-3, (90e-9, 4e-6), 0.8, 150e-6, (120e-9, 100e-9, 93e-9)),
    "Ar": (2e-3, (110e-9, 4e-6), 0.4, 30e-6, (160e-9, 125e-9, 113e-9)),
}

BANDS = ((0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0), (1.0, 1.1))
FLOOR = 1e-6
BAND_TARGET = 1e-4
ACTIVE_TARGET = 1e-5
OFF_AGREEMENT = 1e-12

PLASMA_BAND_TEN_STEP = Study(
    name="plasma_band_ten_step",
    title="The band rule's held-current change over ten steps, against Richardson",
    case={
        "source": "SEP 27 held current: DFT[gJ]/(i omega) - DFT[g'J]/(i omega)**2",
        "run": "prop_capillary(125e-6, L, gas, p, wavelength=800e-9, grid=..., "
        "pulses=..., modes=1, schedule=FixedSchedule.uniform(0, L, L/10), "
        "noise=None, absorber_scheme='split_step', backend='numpy_ref')",
        "helium": "0.8 bar over 5 mm, 150 uJ; grid 90 nm - 4 um; seeds 120, 100, 93 nm",
        "argon": "0.4 bar over 2 mm, 30 uJ; grid 110 nm - 4 um; seeds 160, 125, 113 nm",
        "launch": "a 1.3 fs, 800 nm GaussianPulse and three 4 fs seeds of 2e-4 of "
        "its energy",
        "quantity": "the coarse exit spectrum with the plasma on minus the same off",
        "reference": "Richardson (4 X96 - X48)/3 of the plain trapezoid; its "
        "uncertainty from 24 and 48",
    },
    axis={"name": "band", "unit": "omega/omega_max", "label": "band"},
    columns=(
        Column(
            name="band_rule",
            unit="1 (relative)",
            why="the qualified quantity, a difference of propagated spectra",
            bound=BAND_TARGET,
            rtol=1e-3,
            atol=1e-12,
        ),
        Column(
            name="trapezoid",
            unit="1 (relative)",
            why="today's rule on the same grid, for scale",
            rtol=1e-5,
            atol=1e-12,
        ),
        Column(
            name="reference",
            unit="1 (relative)",
            why="the reference's own uncertainty, from the third refinement",
            rtol=1e-2,
            atol=1e-12,
        ),
        Column(
            name="share",
            unit="1",
            why="the band's share of the active |change|^2",
            rtol=1e-5,
            atol=1e-300,
        ),
    ),
)


def _run(gas, ffac, *, plasma, integration=None):
    """Return the exit spectrum and the run of one ten-step propagation."""
    length, limits, pressure, energy, seeds = CASES[gas]
    # The former geometry: a power-of-two transform nests the refinements' grids,
    # which the Richardson reference needs, and the record was made on it.
    grid = FieldGrid.from_limits(
        length, 800e-9, limits, 400e-15, ffac=ffac, **LEGACY_GRID
    )
    pulses = [GaussianPulse(wavelength=800e-9, fwhm=1.3e-15, energy=energy)] + [
        GaussianPulse(wavelength=seed, fwhm=4e-15, energy=energy * 2e-4)
        for seed in seeds
    ]
    # The reference and the plasma-off baseline are the plain trapezoid's, whatever
    # the default.
    chosen = (integration or "trapezoid") if plasma else None
    with warnings.catch_warnings():
        # The material-range notes of a strong-field run are not under test here.
        warnings.simplefilter("ignore")
        run = prop_capillary(
            125e-6,
            length,
            gas,
            pressure,
            wavelength=800e-9,
            grid=grid,
            pulses=pulses,
            modes=1,
            plasma=plasma,
            schedule=FixedSchedule.uniform(0.0, length, length / 10),
            n_save=2,
            noise=None,
            absorber_scheme="split_step",
            backend="numpy_ref",
            progress="none",
            plasma_integration=chosen,
        )
    return np.asarray(run["field"])[-1, 0, 0], run, grid


@pytest.mark.parametrize("gas", list(CASES))
def test_ten_band_steps_meet_their_targets(request, gas):
    off, _, base = _run(gas, 96.0, plasma=False)
    coarse_off, _, _ = _run(gas, 6.0, plasma=False)
    assert np.max(np.abs(coarse_off - off)) <= OFF_AGREEMENT * np.max(np.abs(off))
    on = {ffac: _run(gas, ffac, plasma=True)[0] for ffac in (6.0, 24.0, 48.0, 96.0)}
    band, run, grid = _run(gas, 6.0, plasma=True, integration="band")
    reference = (4 * on[96.0] - on[48.0]) / 3 - off
    check = (4 * on[48.0] - on[24.0]) / 3 - off
    ratio = grid.omega / grid.omega_max
    active = grid.active_mask
    assert np.array_equal(base.omega[: grid.n_omega], grid.omega)
    total = float(np.linalg.norm(reference[active]) ** 2)

    def row(axis, mask):
        norm = float(np.linalg.norm(reference[mask]))

        def error(change):
            return float(np.linalg.norm(change[mask] - reference[mask])) / norm

        return {
            "series": gas,
            "axis": axis,
            "band_rule": error(band - off),
            "trapezoid": error(on[6.0] - off),
            "reference": error(check),
            "share": norm**2 / total,
        }

    rows = [
        row(
            f"{low:.2f}-{high:.2f}",
            active & (ratio >= low) & ((ratio < high) | (high == BANDS[-1][1])),
        )
        for low, high in BANDS
    ]
    rows.append(row("active", active))
    for one in rows:
        target = ACTIVE_TARGET if one["axis"] == "active" else BAND_TARGET
        if one["axis"] == "active" or one["share"] >= FLOOR:
            assert one["band_rule"] <= target, one
            assert one["reference"] <= target / 10, one
    # The run's own diagnostics, recorded by default under the band rule.
    assert float(np.max(np.asarray(run["stats/plasma_undershoot"]))) <= 1e-3
    assert float(np.max(np.asarray(run["stats/plasma_work_defect"]))) <= 1e-6
    record(PLASMA_BAND_TEN_STEP, request=request, rows=rows, backend="numpy_ref")


# ------------------------------------------------------------- the interface


def _plasma_responses(simulation):
    return [one for one in simulation.responses if isinstance(one, PlasmaResponse)]


@pytest.mark.parametrize("rule", ["band", "trapezoid"])
def test_the_driver_keyword_reaches_every_plasma_response(rule):
    single = build(plasma_integration=rule)
    fill = cast("str", {"He": 0.5, "Ar": 0.5})  # `build` forwards any fill
    mixture = build(gas=fill, plasma_integration=rule)
    for simulation in (single, mixture):
        responses = _plasma_responses(simulation)
        assert responses
        assert {one.integration for one in responses} == {rule}
    # A mixture swap rebuilds the responses from the stored selection, rule and all.
    material = cast("Any", mixture.modes[0]).mixture
    swapped = mixture.with_mixture(
        material.with_densities({material.labels[0]: Constant(4e24)})
    )
    assert {one.integration for one in _plasma_responses(swapped)} == {rule}


def test_the_driver_keyword_is_refused_where_nothing_reads_it():
    with pytest.raises(DriverError, match="envelope plasma converts its currents"):
        build(envelope=True, plasma=True, plasma_integration="band")
    with pytest.raises(DriverError, match="the plasma is off"):
        build(plasma=False, plasma_integration="band")


def test_each_rule_records_itself_and_keys_its_own_resume():
    """Both rules record their response's integration; only the band adds its table."""
    default = build().metadata()
    legacy = build(plasma_integration="trapezoid").metadata()
    band = build(plasma_integration="band").metadata()
    assert band.configuration_hash() == default.configuration_hash()
    assert "plasma_integration" not in legacy.configuration
    records = [
        cast("dict[str, object]", serialise(response))
        for response in cast("list[object]", legacy.configuration["responses"])
    ]
    plasma = [
        one for one in records if str(one["__class__"]).endswith("PlasmaResponse")
    ]
    assert plasma and all(one["integration"] == "trapezoid" for one in plasma)
    recorded = cast("dict[str, Any]", default.configuration["plasma_integration"])
    assert recorded["policy"] == "band"
    assert recorded["half_width"] >= 1
    assert legacy.configuration_hash() != default.configuration_hash()


def test_numba_runs_the_band_rule_and_records_its_diagnostics():
    output = build(plasma_integration="band", backend="numba_cpu").run()
    for name in PLASMA_INTEGRATION_STATISTICS:
        assert name in output.stat_names
    assert float(np.max(np.asarray(output["stats/plasma_undershoot"]))) < 1e-2


def test_native_runs_the_band_rule_and_records_its_diagnostics():
    pytest.importorskip("solprop_native")
    output = build(plasma_integration="band", backend="native").run()
    for name in PLASMA_INTEGRATION_STATISTICS:
        assert name in output.stat_names
    assert float(np.max(np.asarray(output["stats/plasma_undershoot"]))) < 1e-2


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_the_diagnostics_need_a_field_plasma_column(backend):
    simulation = build(
        plasma=False, backend=backend, stats=StatsSpec(names=("plasma_undershoot",))
    )
    with pytest.raises(BackendError, match="carries none"):
        simulation.run()


def test_envelope_plasma_takes_no_running_integral_rule():
    with pytest.raises(MaterialError, match="envelope plasma takes no"):
        responses_for(
            cast("Any", build(plasma=False).modes[0]).mixture,
            plasma=True,
            envelope=True,
            thg=False,
            wavelength=800e-9,
            plasma_integration="band",
        )


def test_a_band_rule_needs_field_plasma_channels():
    problem = build(plasma=False).translate()
    rule = design_running_integral(build().grid, "band")
    with pytest.raises(TranslationError, match="needs field plasma channels"):
        dataclasses.replace(problem.responses, plasma_integral=rule)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The rounding-margin indicator of a noise-seeded mixed run (SEP 36 section 5)."""

import math
from contextlib import ExitStack
from typing import cast

import numpy as np
import pytest

from precision_cases import gnlse, loaded
from solprop.backends.driver import StepDriver
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import native_precisions, select
from solprop.backends.rounding_margin import (
    RoundingMargin,
    occupied_bins,
    rounding_amplitude,
    step_phase,
)
from solprop.driver import Simulation
from solprop.stepper import LowStorageRungeKutta
from solprop.translate.precision import study_admission

U32 = 2.0**-24


def noisy(model: str = "input", **keywords: object) -> Simulation:
    """Return the shared soliton fibre, mixed and seeded with a noise model."""
    return gnlse("mixed", noise=model, seed=3, **keywords)  # pyright: ignore[reportArgumentType]


def margin_record(backend: object) -> dict[str, object]:
    """Return the indicator's record from a backend's metadata."""
    record = cast(StepDriver[object], backend).metadata()["rounding_margin"]
    return cast(dict[str, object], record)


def planes_of(record: dict[str, object]) -> list[dict[str, object]]:
    """Return the record's per-plane entries."""
    return cast(list[dict[str, object]], record["planes"])


def test_the_amplitude_is_the_section_5_formula():
    assert rounding_amplitude(4.0, 1024, 256, c_s=2.0) == pytest.approx(
        2.0 * U32 * 10.0 * math.sqrt(4.0 * 256 / 1024)
    )
    assert occupied_bins(np.zeros(4)).tolist() == [False] * 4
    # -30 dB of the peak is the threshold, inclusive.
    assert occupied_bins(np.array([1.0, 1e-3, 0.99e-3])).tolist() == [True, True, False]


def test_only_accepted_steps_with_a_finite_phase_count():
    margin = RoundingMargin(None)
    margin.reset(0.0)
    margin.accept(0.1, 0.01)
    margin.accept(math.nan, 0.02)  # a zero state norm adds nothing
    margin.accept(0.2, 0.03)
    assert margin.sum_phi_squared == pytest.approx(0.05)
    state = np.zeros((1, 1, 1, 8), dtype=complex)
    margin.plane(0.03, state, state)
    (entry,) = planes_of(margin.record())
    assert entry["sigma_r"] is None
    assert entry["absent"] == "zero state norm"
    margin.accept(None)
    assert not margin.active
    assert "only the Runge-Kutta march" in str(margin.record()["absent"])


def test_a_restore_takes_the_sum_at_the_resume_position():
    margin = RoundingMargin(None)
    margin.reset(0.0)
    margin.accept(0.1, 0.01)
    margin.plane(0.01, np.ones((1, 4), dtype=complex), np.ones((1, 4), dtype=complex))
    margin.accept(0.1, 0.02)
    record = margin.record()
    for z, expected in ((0.01, 0.01), (0.02, 0.02)):
        resumed = RoundingMargin(None)
        resumed.reset(z)
        assert resumed.restore(record, z)
        assert resumed.sum_phi_squared == pytest.approx(expected)
    lost = RoundingMargin(None)
    lost.reset(0.015)
    assert not lost.restore(record, 0.015)
    assert "without the running sum" in str(lost.record()["absent"])


@pytest.mark.parametrize("model", ["input", "modified"])
def test_a_noise_seeded_mixed_run_records_the_indicator(model):
    with study_admission(), ExitStack() as cleanup:
        simulation = noisy(model)
        backend = loaded(simulation, cleanup)
        backend.propagate_to(0.05)
        observation = backend.observe()
        record = margin_record(backend)
    assert "heuristic" in str(record["label"])
    assert "gain and coherent bias" in str(record["label"])
    assert record["c_s"] == 1.0
    assert "absent" not in record
    (entry,) = planes_of(record)
    assert entry["z"] == pytest.approx(0.05)
    total = cast(float, record["sum_phi_squared"])
    assert total > 0.0
    field = observation.field
    assert field is not None
    energy = (np.abs(field) ** 2).reshape(-1, field.shape[-1]).sum(axis=0)
    n_occupied = int(occupied_bins(energy).sum())
    assert entry["n_occupied"] == n_occupied
    sigma_r = cast(float, entry["sigma_r"])
    n = field.shape[-1]
    assert sigma_r == pytest.approx(
        U32 * math.log2(n) * math.sqrt(total * n_occupied / n)
    )
    sigma_q = cast(float, entry["sigma_q"])
    assert 0.0 < sigma_q < 1.0
    assert entry["margin_db"] == pytest.approx(20.0 * math.log10(sigma_q / sigma_r))


def test_each_accepted_step_adds_its_last_stage_phase():
    """``phi = dz ||k_last|| / ||A||`` in the controller's in-band norm, per step."""
    with study_admission(), ExitStack() as cleanup:
        backend = cast(NumpyRefBackend, loaded(noisy(), cleanup))
        phases = []
        original = backend._trial_step

        def watched(z: float, dz: float):
            anchor = backend._require_state().copy()
            result = original(z, dz)
            mask = backend._problem.grids.norm_mask
            phases.append(step_phase(anchor, backend._last_stages[-1], dz, mask))
            return result

        backend._trial_step = watched
        backend.propagate_to(0.05)
        record = margin_record(backend)
    assert len(phases) == 10
    assert record["sum_phi_squared"] == pytest.approx(sum(p * p for p in phases))


def test_an_adaptive_run_counts_its_accepted_steps_only():
    with study_admission(), ExitStack() as cleanup:
        backend = cast(NumpyRefBackend, loaded(noisy(steps=None), cleanup))
        backend._step_phase = lambda dz: 1.0  # one per accepted step
        backend.propagate_to(0.05)
        history = backend.observe().step_history
        record = margin_record(backend)
    assert history.n_rejected >= 0
    assert record["sum_phi_squared"] == history.n_steps - history.n_rejected


def test_the_indicator_is_absent_without_a_noise_model_or_a_runge_kutta_march():
    with ExitStack() as cleanup:
        quiet = loaded(gnlse("mixed"), cleanup)
        quiet.propagate_to(0.05)
        quiet.observe()
        assert margin_record(quiet) == {
            "label": margin_record(quiet)["label"],
            "c_s": 1.0,
            "absent": "the run has no noise model",
        }
        double = loaded(gnlse("double"), cleanup)
        assert "rounding_margin" not in cast(StepDriver[object], double).metadata()
        with study_admission():
            register = loaded(noisy(integrator=LowStorageRungeKutta()), cleanup)
        register.propagate_to(0.05)
        register.observe()
        record = margin_record(register)
    assert "only the Runge-Kutta march" in str(record["absent"])
    assert record["planes"] == []


def test_a_reused_runner_starts_afresh_and_a_resume_continues():
    with study_admission(), ExitStack() as cleanup:
        simulation = noisy()
        problem = simulation.translate()
        whole = loaded(simulation, cleanup)
        whole.propagate_to(0.02)
        middle = whole.observe().field
        assert middle is not None
        saved = cast(StepDriver[object], whole).metadata()
        whole.propagate_to(0.05)
        whole.observe()
        uninterrupted = margin_record(whole)["sum_phi_squared"]
        whole.set_initial(problem.initial.field0, None)
        assert margin_record(whole)["sum_phi_squared"] == 0.0
        assert planes_of(margin_record(whole)) == []

        resumed = select("numpy_ref", problem, grid=simulation.grid)
        cleanup.callback(resumed.close)
        resumed.set_initial(middle, None, z=0.02)
        cast(StepDriver[object], resumed).restore_rounding_margin(saved)
        resumed.propagate_to(0.05)
        resumed.observe()
        continued = margin_record(resumed)
    assert "absent" not in continued
    assert continued["sum_phi_squared"] == pytest.approx(uninterrupted, rel=1e-12)


@pytest.mark.skipif(
    "mixed" not in native_precisions(), reason="no native build runs mixed precision"
)
@pytest.mark.parametrize("model", ["input", "modified"])
def test_native_mixed_records_the_oracles_indicator(model):
    """Class R: the same steps on both, so the sums agree to the mixed source bound."""
    sums = {}
    with study_admission(), ExitStack() as cleanup:
        for name in ("numpy_ref", "native"):
            simulation = noisy(model, backend=name)
            problem = simulation.translate()
            backend = select(name, problem, grid=simulation.grid)
            cleanup.callback(backend.close)
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(0.05)
            backend.observe()
            record = margin_record(backend)
            assert "absent" not in record, record
            sums[name] = cast(float, record["sum_phi_squared"])
            (entry,) = planes_of(record)
            assert cast(float, entry["margin_db"]) > 0.0
    assert sums["native"] == pytest.approx(sums["numpy_ref"], rel=1e-6)

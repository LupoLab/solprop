# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ``tdse=`` capillary entry point and its defaults (SEP 20 items 15 and 16)."""

import dataclasses
import inspect
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
from tests.driver.test_sep12_sep14_integration import _Binding
from tests.hacc_cases import hacc_response, plain, preset_basis

import solprop.driver.simulation as simulation_module
from record_numbers import assert_record_numbers, record_skeleton, skeleton_hash
from solprop.driver import Simulation, prop_capillary
from solprop.driver import tdse as runtime
from solprop.driver.simulation import SimulationSettings
from solprop.exceptions import (
    DriverError,
    MaterialRangeWarning,
    TDSECostWarning,
    TranslationError,
)
from solprop.nonlinear.tdse import (
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.output import MemoryOutput
from solprop.output.schema import serialise
from solprop.stats import StatsSpec
from solprop.translate.tdse_source import TDSE_UNAVAILABLE_STATISTICS

_LENGTH = 10 * 2.0**-20


def _sae():
    basis = TDSEBasis("entry", (0.0, 1e-9), (8,), 5, 1e-16, source="entry fixture")
    calibration = TDSECalibration(
        "entry",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="manufactured",
        limitations=("no physical qualification",),
    )
    return TDSEGasResponse(
        "helium", basis, 1e-19, "tdse_native", 1.0, calibration, TDSESampling()
    )


def _capillary(**changes):
    values = {
        "lambda0": 800e-9,
        "wavelength_limits": (400e-9, 1.6e-6),
        "trange": 60e-15,
        "fwhm": 6e-15,
        "energy": 1e-12,
        "modes": 1,
        "n_r": 4,
        "noise": None,
        "dz_init": 2.0**-20,
        "dz_max": 2.0**-20,
    }
    values.update(changes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(256e-6, _LENGTH, "He", 0.1, **values)


def test_tdse_keyword_builds_the_one_response_run_in_one_step(caplog):
    response = _sae()
    execution = TDSEExecution(max_atomic_calls=7)
    with caplog.at_level(logging.INFO, logger="solprop.driver.simulation"):
        simulation = _capillary(tdse=response, tdse_execution=execution)
    assert simulation.responses == (response,)
    assert simulation.settings.tdse_execution == execution
    assert simulation.settings.backend == "numpy_ref"
    assert "resolves to numpy_ref" in caplog.text
    record = plain(simulation.configuration["tdse_entry"])
    assert record["backend"] == {"requested": "auto", "resolved": "numpy_ref"}
    assert record["model"] == "sae"
    assert simulation.translate().tdse is not None
    # A default execution record is taken when none is given.
    assert _capillary(tdse=response).settings.tdse_execution == TDSEExecution()


def test_hacc_window_duration_resolves_from_the_capillary_request():
    simulation = _capillary(tdse=hacc_response(preset_basis(1), atomic_window="auto"))
    (declared,) = simulation.responses
    assert isinstance(declared, TDSEGasResponse)
    assert declared.atomic_window == TDSEAtomicWindow(duration=60e-15)
    entry = plain(simulation.configuration["tdse_entry"])
    assert entry["atomic_window_duration"] == "trange"
    explicit = _capillary(
        tdse=hacc_response(preset_basis(1), atomic_window=TDSEAtomicWindow(60e-15))
    )
    assert (
        explicit.translate().tdse.identity == simulation.translate().tdse.identity  # type: ignore[union-attr]
    )
    with pytest.raises(DriverError, match="unresolved"):
        _capillary(
            tdse=hacc_response(preset_basis(1), atomic_window="auto"),
            trange=None,
            grid=simulation._base_grid,
            wavelength_limits=None,
        )


def test_lower_level_construction_refuses_an_unresolved_window_before_any_provider(
    monkeypatch,
):
    parent = _capillary(tdse=_sae())

    def unexpected():
        pytest.fail("the provider was loaded")

    monkeypatch.setattr(runtime, "_load_provider", unexpected)
    simulation = Simulation(
        parent.grid,
        parent.modes,
        [hacc_response(preset_basis(1), atomic_window="auto")],
        parent._field0,
        settings=parent.settings,
        noise=None,
        profiles=parent.profiles,
        frame=parent.frame,
    )
    with pytest.raises(TranslationError, match="unresolved"):
        simulation.translate()


@pytest.mark.parametrize(
    "keyword",
    [
        {"plasma": True},
        {"plasma": False},
        {"raman": False},
        {"preionisation": 0.0},
        {"plasma_integration": "band"},
        {"ionisation_field_range": (1.0, 2.0)},
        {"thg": True},
        {"kerr": False},
    ],
)
def test_electronic_keywords_beside_tdse_are_refused_by_name(keyword):
    (name,) = keyword
    with pytest.raises(DriverError, match=name):
        _capillary(tdse=_sae(), **keyword)


def test_tdse_execution_alone_and_a_foreign_record_are_refused():
    with pytest.raises(DriverError, match="tdse_execution requires"):
        _capillary(tdse_execution=TDSEExecution())
    # The record's own check, beneath the test session's annotation checker.
    entry = inspect.unwrap(simulation_module._tdse_entry)
    with pytest.raises(DriverError, match="must be a TDSEGasResponse"):
        entry(
            object(),
            None,
            trange=None,
            kerr=True,
            unset={},
            backend="auto",
            step_planes=None,
            save_planes=None,
            n_save=None,
        )


def test_explicit_incompatible_settings_keep_their_existing_refusals():
    with pytest.raises(DriverError, match="numpy_ref and Multistep"):
        _capillary(tdse=_sae(), backend="numba_cpu")


def test_statistics_default_excludes_what_a_tdse_response_cannot_supply():
    spec = _capillary(tdse=_sae()).settings.stats
    assert spec is not None
    names = spec.names
    assert names and not set(names) & TDSE_UNAVAILABLE_STATISTICS
    explicit = StatsSpec(names=("energy",))
    assert _capillary(tdse=_sae(), stats=explicit).settings.stats == explicit
    with pytest.raises(TranslationError):
        _capillary(
            tdse=_sae(), stats=StatsSpec(names=("energy", "electron_density"))
        ).translate()


def test_tdse_output_defaults_and_their_explicit_overrides():
    final_only = _capillary(tdse=_sae())
    assert final_only.settings.step_planes == 1
    assert tuple(final_only.settings.save_planes) == (_LENGTH,)
    entry = plain(final_only.configuration["tdse_entry"])
    assert entry["requested_planes"] == "final_only"
    assert entry["step_planes"] == {"requested": None, "resolved": 1}
    no_steps = _capillary(tdse=_sae(), step_planes=0)
    assert no_steps.settings.step_planes == 0
    planes = _capillary(tdse=_sae(), n_save=5)
    assert len(planes.settings.save_planes) == 4 and planes.settings.step_planes == 1
    explicit = _capillary(tdse=_sae(), save_planes=[_LENGTH / 2, _LENGTH])
    assert tuple(explicit.settings.save_planes) == (_LENGTH / 2, _LENGTH)
    assert plain(explicit.configuration["tdse_entry"])["requested_planes"] == (
        "explicit"
    )
    # The lower-level settings keep their defaults.
    assert SimulationSettings.__dataclass_fields__["step_planes"].default == 0


# Computed on the base commit of SEP 20 Phase 1: a rate-based capillary call without
# `tdse` builds this exact configuration, responses, itinerary and statistics. A change
# of optmat's models would legitimately move it; recompute it there, not here. Until
# SEP 25 one SHA-256 pinned the whole record, last bits of its derived numbers
# included, which another platform's libm and FFT move. Its structure is still hashed
# exactly; its numbers are stored beside this test (recorded from the state the
# former digest certified) and compared within a bound (tests/record_numbers.py).
_UNCHANGED = "aad63c25bc485771c6e6cf1cceb7868f877a5de6a7fd5dc07c741b68cdb50683"


def test_without_tdse_a_representative_call_is_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            265e-6,
            0.05,
            "He",
            (0.005, 0.15),
            lambda0=800e-9,
            wavelength_limits=(9e-8, 1.6e-6),
            trange=60e-15,
            fwhm=10e-15,
            energy=1e-4,
            modes=1,
            backend="numpy_ref",
        )
    stats = simulation.settings.stats
    assert stats is not None
    record = {
        "configuration": serialise(simulation.configuration),
        "responses": serialise(list(simulation.responses)),
        "save_planes": [float(x) for x in simulation.settings.save_planes],
        "step_planes": simulation.settings.step_planes,
        "backend": simulation.settings.backend,
        "stats": list(stats.names),
        "tdse_execution": simulation.settings.tdse_execution,
    }
    skeleton, numbers, floors = record_skeleton(
        json.loads(json.dumps(record, sort_keys=True, default=repr))
    )
    assert skeleton_hash(skeleton) == _UNCHANGED
    assert_record_numbers(
        numbers,
        floors,
        Path(__file__).with_suffix("")
        / "test_without_tdse_a_representative_call_is_unchanged.npz",
    )
    assert simulation.settings.step_planes == 0
    assert len(simulation.settings.save_planes) == 200


def test_prop_capillary_forwards_the_tdse_keywords(monkeypatch):
    seen = []

    def capture(self, output=None, **kwargs):
        seen.append(self)
        return MemoryOutput()

    monkeypatch.setattr(Simulation, "run", capture)
    response = _sae()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        prop_capillary(
            256e-6,
            _LENGTH,
            "He",
            0.1,
            lambda0=800e-9,
            wavelength_limits=(400e-9, 1.6e-6),
            trange=60e-15,
            fwhm=6e-15,
            energy=1e-12,
            n_r=4,
            noise=None,
            tdse=response,
            tdse_execution=TDSEExecution(max_batch_rows=2),
        )
    (simulation,) = seen
    assert simulation.responses == (response,)
    assert simulation.settings.tdse_execution == TDSEExecution(max_batch_rows=2)
    assert simulation.settings.step_planes == 1


def _run_with_zero_binding(monkeypatch, **changes):
    bindings = []

    def prepare(descriptor, execution, *args, **kwargs):
        binding = _Binding(descriptor, execution.max_atomic_calls)
        bindings.append(binding)
        return binding

    monkeypatch.setattr(runtime, "prepare_binding", prepare)
    simulation = _capillary(tdse=_sae(), stats=StatsSpec(names=("energy",)), **changes)
    simulation.run(progress="none")
    return bindings


@pytest.mark.medium
def test_a_many_node_rule_warns_once_with_the_atoms_per_call_reason(monkeypatch):
    with pytest.warns(TDSECostWarning, match=r"5 nodes.*one atom") as record:
        _run_with_zero_binding(monkeypatch, n_r=4)
    assert len([one for one in record if one.category is TDSECostWarning]) == 1


@pytest.mark.medium
def test_a_field_scale_rule_raises_no_cost_warning(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("error", TDSECostWarning)
        bindings = _run_with_zero_binding(monkeypatch, radial_rule="field_scale", n_r=2)
    assert bindings and bindings[0].closed
    assert np.all(np.isfinite(bindings[0].inputs[0]))


def test_settings_replacement_keeps_tdse_resources_after_translation():
    simulation = _capillary(tdse=_sae())
    simulation.translate()
    simulation.settings = dataclasses.replace(
        simulation.settings, tdse_execution=TDSEExecution(max_atomic_calls=3)
    )
    assert simulation.settings.tdse_execution == TDSEExecution(max_atomic_calls=3)

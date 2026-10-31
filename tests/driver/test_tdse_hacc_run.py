# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run haCC end to end through the real binding and a fake provider (SEP 20 item 17).

The fake's first-order current is the manufactured linear response
``alpha_optmat / k1`` and its remainder an exact polynomial, so the consistency
record, the ledger and the source are all known in closed form. This is the haCC
checkpoint-resume test; the real haCC segments are not deliberately interrupted.
"""

import dataclasses
import json
import warnings
from typing import Any

import numpy as np
import pytest
from tests.hacc_cases import (
    alpha_law,
    hacc_response,
    install,
    plain,
    preset_basis,
    tiny_basis,
)

from solprop.driver import Simulation
from solprop.driver.tdse_consistency import ConsistencyResult
from solprop.exceptions import (
    DriverError,
    MaterialRangeWarning,
    OutputError,
    ResponseBudgetError,
    TDSEConsistencyError,
    TDSEConsistencyWarning,
    TDSEDomainWarning,
    TDSEUnqualifiedModelWarning,
)
from solprop.nonlinear.tdse import TDSEConsistency, TDSEExecution
from solprop.output import MemoryOutput, open_file, resume
from solprop.output.schema import SCHEMA_VERSION
from solprop.stats import StatsSpec

pytestmark = pytest.mark.medium

_LENGTH = 6 * 2.0**-20


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache" / "tdse-consistency"


@pytest.fixture
def fake(monkeypatch):
    fake = install(monkeypatch)
    # A 2% linear model error and a dyadic fifth-order remainder. The field is about
    # 1e6 V/m, so 2^-200 E^5 ~ 1e-30 C m/s stays ten orders below the linear current.
    fake.response = alpha_law(1.02, lambda5=2.0**-200)
    return fake


def _simulation(basis=None, *, limit=1000, **changes):
    response = hacc_response(
        preset_basis(1) if basis is None else basis,
        atomic_window=changes.pop("atomic_window", "auto"),
        consistency=changes.pop("consistency", TDSEConsistency()),
    )
    values = {
        "wavelength": 800e-9,
        "wavelength_limits": (400e-9, 1.6e-6),
        "trange": 40e-15,
        "fwhm": 6e-15,
        "energy": 1e-12,
        "modes": 1,
        "radial_rule": "field_scale",
        "n_r": 2,
        "noise": None,
        "dz_init": 2.0**-20,
        "dz_max": 2.0**-20,
        "stats": StatsSpec(names=("energy",)),
        "tdse": response,
        "tdse_execution": TDSEExecution(max_atomic_calls=limit),
    }
    values.update(changes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(
            256e-6, _LENGTH, "He", 0.1, **{"noise": None, **values}
        )


def _run(simulation, output=None, **kwargs) -> Any:
    """Run quietly; the returned handler is read for both its planes and outcome."""
    with warnings.catch_warnings():
        # The preset's tested window is far longer than this short test window.
        warnings.simplefilter("ignore", TDSEDomainWarning)
        warnings.simplefilter("ignore", TDSEUnqualifiedModelWarning)
        return simulation.run(output, progress="none", **kwargs)


def _execution(output):
    assert output.outcome is not None
    return output.outcome.execution["tdse"]


@pytest.mark.parametrize("window", ["auto", None], ids=["window", "full"])
def test_hacc_run_records_model_window_consistency_and_ledger(fake, cache, window):
    simulation = _simulation(atomic_window=window)
    output = _run(simulation)
    assert output.metadata is not None
    assert output.metadata.schema_version == SCHEMA_VERSION == 1
    descriptor = json.loads(plain(output.metadata.tdse)["configuration_json"])
    assert descriptor["schema"] == "solprop.tdse-descriptor.v5"
    assert descriptor["response"]["model"] == "hacc"
    assert descriptor["response"]["basis"]["preset"] == preset_basis(1).preset
    record = _execution(output)
    assert record["model"] == "hacc"
    assert record["hacc"]["closure"] is True and record["hacc"]["custom"] is False
    assert ("atomic_window" in record) is (window == "auto")
    consistency = record["consistency"]
    assert consistency["outcome"] == "passed" and consistency["from_cache"] is False
    assert consistency["reduction"]["deviation"] == pytest.approx(0.02, abs=1e-9)
    assert len(list(cache.glob("*.json"))) == 1
    calls = plain(output.outcome.atomic_calls)
    assert calls["diagnostic_calls"] == 1 and calls["failed_calls"] == 0
    logical = plain(output.outcome.logical_calls)
    assert calls["attempted"] == 1 + logical["startup"] + logical["predictor"]
    chunk = record["last_call"]["chunks"][0]
    assert chunk["neglected_current_bound"][0]["zero_reference"] is False
    assert np.all(np.isfinite(output.fields(convention="fft")))
    # Every accepted step is saved, and the requested final position.
    assert output.n_saved == 1 + logical["accepted_steps"]


def test_warned_and_skipped_checks_record_their_outcome(fake):
    fake.response = alpha_law(1.2)
    with pytest.warns(TDSEConsistencyWarning, match="kappa_1") as record:
        output = _run(_simulation())
    assert len([one for one in record if one.category is TDSEConsistencyWarning]) == 1
    assert _execution(output)["consistency"]["outcome"] == "warned"
    with pytest.warns(TDSEConsistencyWarning, match="skipped") as record:
        output = _run(_simulation(consistency=TDSEConsistency(enforce=False)))
    assert len([one for one in record if one.category is TDSEConsistencyWarning]) == 1
    assert _execution(output)["consistency"]["outcome"] == "skipped"
    assert plain(output.outcome.atomic_calls)["diagnostic_calls"] == 0


@pytest.mark.parametrize("law", ["refused", "invalid"])
def test_refused_check_stops_before_any_propagation_call(fake, law):
    if law == "refused":
        fake.response = alpha_law(0.5)
    else:
        fake.response = lambda fields, spacing: (
            np.zeros_like(fields),
            np.zeros_like(fields),
        )
    output = MemoryOutput()
    with pytest.raises(TDSEConsistencyError, match=law) as error:
        _run(_simulation(), output)
    result = error.value.result
    assert isinstance(result, ConsistencyResult) and result.outcome == law
    # Only the one-row diagnostic reached the provider.
    assert [call.shape[0] for call in fake.calls] == [1]
    assert output.n_saved == 0 and output.terminal_checkpoint is None
    assert output.metadata is not None
    assert output.metadata.backend is not None
    recorded = plain(output.metadata.backend.settings)["tdse"]["consistency"]
    assert recorded["outcome"] == law


def test_custom_basis_is_checked_and_recorded_as_unqualified(fake):
    with pytest.warns(TDSEUnqualifiedModelWarning):
        output = _simulation(tiny_basis()).run(progress="none")
    record = _execution(output)
    assert record["hacc"]["custom"] is True and record["hacc"]["qualified"] is False
    assert record["consistency"]["outcome"] == "passed"
    assert record["consistency"]["preset"] is None


def test_checkpoint_and_resume_keep_identities_and_never_repeat_the_check(
    fake, tmp_path
):
    path = tmp_path / "hacc.h5"
    simulation = _simulation(limit=40)
    simulation.settings = dataclasses.replace(simulation.settings, resume_history=False)
    with pytest.raises(ResponseBudgetError):
        _run(simulation, path)
    first = resume(path)
    assert first.outcome is not None and first.outcome.status == "budget_exhausted"
    assert first.outcome.atomic_calls["diagnostic_calls"] == 1
    with open_file(path) as saved:
        assert saved.metadata is not None
        key = saved.metadata.configuration_hash()
    calls_before = len(fake.calls)
    simulation.settings = dataclasses.replace(
        simulation.settings, tdse_execution=TDSEExecution(max_atomic_calls=1000)
    )
    _run(simulation, path, resume=True)
    final = resume(path)
    assert final.outcome is not None and final.outcome.status == "complete"
    with open_file(path) as saved:
        assert saved.metadata is not None
        assert saved.metadata.configuration_hash() == key
    # The cached check is reused: no second diagnostic row reached the provider.
    assert final.outcome.atomic_calls["diagnostic_calls"] == 1
    assert all(
        call.shape[0] != 1 or call.shape[1] != 1601
        for call in fake.calls[calls_before:]
    )
    consistency = plain(final.outcome.execution)["tdse"]["consistency"]
    assert consistency["from_cache"] is True
    # Without a kept Adams history the resumed march charges a fresh startup.
    assert final.outcome.logical_calls["startup"] == 2
    # A resume against a different basis is refused by its identity.
    other = _simulation(preset_basis(2))
    with pytest.raises(OutputError):
        _run(other, path, resume=True)


def test_a_hacc_run_needs_a_binding_with_a_diagnostic_row(fake, monkeypatch):
    from tests.driver.test_sep12_sep14_integration import _Binding  # noqa: PLC0415

    from solprop.driver import tdse as runtime  # noqa: PLC0415

    monkeypatch.setattr(
        runtime,
        "prepare_binding",
        lambda descriptor, execution, *args, **kwargs: _Binding(descriptor, None),
    )
    with pytest.raises(DriverError, match="diagnostic row"):
        _run(_simulation())


def test_a_failed_refusal_record_keeps_the_consistency_error_primary(
    fake, monkeypatch, caplog
):
    import solprop.driver.simulation as simulation_module  # noqa: PLC0415

    def broken(*args, **kwargs):
        raise OutputError("disk full")

    monkeypatch.setattr(simulation_module, "open_run_record", broken)
    fake.response = alpha_law(0.5)
    with pytest.raises(TDSEConsistencyError, match="refused"):
        _run(_simulation())
    assert "could not record the refused consistency check" in caplog.text


def test_the_run_closes_its_diagnostic_plan_before_the_propagation_plans(fake):
    output = _run(_simulation())
    assert _execution(output)["consistency"]["outcome"] == "passed"
    enters = [event for event in fake.events if event.startswith("enter-plan")]
    closes = [event for event in fake.events if event.startswith("close-plan")]
    diagnostic = "plan-1-1601"
    assert enters[0] == f"enter-{diagnostic}"
    assert fake.events.index(f"close-{diagnostic}") < fake.events.index(enters[1])
    assert closes[0] == f"close-{diagnostic}"


def test_the_gate_leaves_an_already_built_binding_as_it_is(fake, monkeypatch):
    from solprop.driver import tdse as runtime  # noqa: PLC0415

    prepare = runtime.prepare_binding

    def eager(*args, **kwargs):
        kwargs["defer_plans"] = False
        return prepare(*args, **kwargs)

    monkeypatch.setattr(runtime, "prepare_binding", eager)
    output = _run(_simulation())
    assert _execution(output)["consistency"]["outcome"] == "passed"
    assert output.outcome.status == "complete"

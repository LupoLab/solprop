# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Save every accepted TDSE step at no extra call (SEP 20 section 7.1 item 20).

The ``tdse=`` default saves each accepted endpoint beside a final-only itinerary.
Against ``step_planes=0`` on the same itinerary the march is the same: within one run
its saves are its accepted endpoints and its ledger counts one provider evaluation per
attempted step and nothing more, exactly. Two runs are adaptive, so across them the
work is a band and the final field an accuracy envelope against the same case at a
hundredth of the tolerance (SEP 25 section 4.4, ``tests/march_bounds.py``).
"""

import warnings
from typing import cast

import numpy as np
import pytest
from tests.hacc_cases import alpha_law, hacc_response, install, plain, preset_basis

from bounds import assert_work_within
from march_bounds import TIGHTER, assert_in_envelope
from solprop.driver import Simulation
from solprop.exceptions import (
    GasFlowWarning,
    MaterialRangeWarning,
    TDSEDomainWarning,
)
from solprop.log import NullReporter
from solprop.nonlinear.tdse import TDSEExecution
from solprop.output.hdf5 import open_file, resume
from solprop.stats import StatsSpec
from solprop.stepper import PIController

pytestmark = pytest.mark.medium

_LENGTH = 6 * 2.0**-20

WORK_BAND = 1.1
"""How far apart two adaptive runs' step and call counts may be: a fork's few steps."""


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def fake(monkeypatch):
    fake = install(monkeypatch)
    fake.response = alpha_law(1.0, lambda5=2.0**-200)
    return fake


def _simulation(**changes):
    values = {
        "wavelength": 800e-9,
        "wavelength_limits": (400e-9, 1.6e-6),
        "trange": 40e-15,
        "fwhm": 6e-15,
        "energy": 1e-12,
        "radial_rule": "field_scale",
        "n_r": 2,
        "noise": None,
        "dz_init": 2.0**-20,
        "dz_max": 2.0**-20,
        "stats": StatsSpec(names=("energy",)),
        "tdse": hacc_response(preset_basis(1), atomic_window="auto"),
        "tdse_execution": TDSEExecution(max_atomic_calls=10_000),
    }
    values.update(changes)
    # A pressure stage at mid-length puts a kink in the density path.
    pressure = ([0.0, _LENGTH / 2, _LENGTH], [0.1, 0.2, 0.1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        # A micrometre-long manufactured fill is far outside the laminar regime.
        warnings.simplefilter("ignore", GasFlowWarning)
        return Simulation.from_capillary(
            256e-6,
            _LENGTH,
            "He",
            pressure,
            vacuum_end="floor",
            **{"noise": None, **values},
        )


def _run(simulation, output=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEDomainWarning)
        return simulation.run(output, progress=kwargs.pop("progress", "none"), **kwargs)


def assert_no_extra_call(outcome):
    """Assert a run's ledger counts one evaluation per attempted step, exactly.

    Every attempted step, accepted or rejected, makes one predictor evaluation; the
    provider's atomic calls are those, the startup's and the consistency diagnostic's.
    """
    logical, atomic = plain(outcome.logical_calls), plain(outcome.atomic_calls)
    assert logical["predictor"] == logical["accepted_steps"] + logical["rejected_steps"]
    assert atomic["attempted"] == (
        logical["startup"] + logical["predictor"] + atomic["diagnostic_calls"]
    )


def assert_same_work(one, other):
    """Assert two adaptive runs' work agrees within :data:`WORK_BAND`."""
    first, second = plain(one.logical_calls), plain(other.logical_calls)
    for name in ("accepted_steps", "predictor"):
        assert_work_within(second[name], first[name], factor=WORK_BAND, what=name)
    assert_work_within(
        plain(other.atomic_calls)["attempted"],
        plain(one.atomic_calls)["attempted"],
        factor=WORK_BAND,
        what="atomic calls",
    )


def reference_field(monkeypatch, tmp_path):
    """Return the final field of the case at a hundredth of its tolerance."""
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "cache-reference"))
    rtol = cast("PIController", _simulation().settings.controller).rtol
    return _run(_simulation(rtol=rtol / TIGHTER)).fields(convention="fft")[-1]


def _accepted_ends(output):
    ends = []
    for plane in output.planes():
        history = plane.step_history
        ends += list((history.z + history.h)[history.accepted])
    return np.array(ends)


def _rejected_ends(output):
    ends = []
    for plane in output.planes():
        history = plane.step_history
        ends += list((history.z + history.h)[~history.accepted])
    return np.array(ends)


def test_natural_saves_are_the_same_march_at_no_extra_call(fake, monkeypatch, tmp_path):
    runs = {}
    for steps in (0, 1):
        # A fresh cache for each run, so each pays its own consistency check.
        monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / f"cache-{steps}"))
        before = len(fake.calls)
        output = _run(_simulation(step_planes=steps))
        runs[steps] = (output, len(fake.calls) - before)
    bare, natural = runs[0][0], runs[1][0]
    assert bare.outcome is not None and natural.outcome is not None
    # No extra call: each run's ledger is its own attempted steps, exactly, and the
    # provider saw what the ledger counts.
    for output, provider_calls in runs.values():
        assert output.outcome is not None
        assert_no_extra_call(output.outcome)
        assert plain(output.outcome.atomic_calls)["attempted"] == provider_calls
    assert_same_work(bare.outcome, natural.outcome)
    assert natural.outcome.logical_calls["rejected_steps"] >= 1
    assert_in_envelope(
        [natural.fields(convention="fft")[-1]],
        reference_field(monkeypatch, tmp_path),
        bare.fields(convention="fft")[-1],
        what="the final field",
    )
    z = np.asarray(natural["z"])
    # Every accepted endpoint once, the requested end not twice, no rejected trial.
    np.testing.assert_array_equal(z[1:], _accepted_ends(natural))
    assert np.all(np.diff(z) > 0) and z[-1] == _LENGTH
    assert not set(_rejected_ends(natural)) - set(_accepted_ends(natural)) & set(z)
    assert np.asarray(bare["z"]).tolist() == [0.0, _LENGTH]
    assert natural.metadata is not None
    assert natural.metadata.stepper.limits.step_planes == 1


def test_the_density_kink_is_saved_once_when_a_step_lands_on_it(fake):
    output = _run(_simulation())
    z = np.asarray(output["z"])
    landed = np.isclose(z, _LENGTH / 2, rtol=1e-12, atol=0.0)
    # The stage break is a physical discontinuity the march lands on; the save there
    # is the accepted endpoint itself, recorded once.
    assert landed.sum() == 1
    assert len(np.unique(z)) == len(z)


class _Stop(NullReporter):
    def __init__(self, after):
        self.after = after
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen >= self.after:
            raise RuntimeError("stopped by the callback")


def test_a_callback_interruption_resumes_from_its_saved_endpoint(fake, tmp_path):
    whole = _run(_simulation())
    path = tmp_path / "natural.h5"
    with pytest.raises(RuntimeError, match="callback"):
        _run(_simulation(), path, progress=_Stop(5))
    with open_file(path) as saved:
        z_stopped = np.asarray(saved["z"]).copy()
    stopped = resume(path)
    # The entrance is saved before the first callback, then five accepted steps.
    assert stopped.n_saved == 6
    assert stopped.outcome is not None
    calls_at_stop = plain(stopped.outcome.atomic_calls)["attempted"]
    _run(_simulation(), path, resume=True)
    finished = resume(path)
    assert finished.plane.z == _LENGTH
    assert finished.outcome is not None and finished.outcome.status == "complete"
    with open_file(path) as saved:
        z = np.asarray(saved["z"])
    assert np.all(np.diff(z) > 0)
    # The planes saved before the stop are kept as they were, exactly.
    np.testing.assert_array_equal(z[:6], z_stopped)
    assert z_stopped.size == 6 and whole.n_saved > 6
    # The continuation starts at the saved accepted endpoint and appends after it.
    assert z[6] > stopped.plane.z == z[5]
    assert len(np.unique(z)) == len(z)
    # Charges are cumulative across the interruption. The uninterrupted run seeded
    # the consistency cache, so neither execution of this file paid for the check,
    # and the provider saw exactly one diagnostic row in the whole test.
    outcome = finished.outcome
    assert plain(outcome.atomic_calls)["attempted"] > calls_at_stop
    assert outcome.atomic_calls["diagnostic_calls"] == 0
    assert sum(call.shape == (1, 1601) for call in fake.calls) == 1
    assert outcome.logical_calls["startup"] in (1, 2)

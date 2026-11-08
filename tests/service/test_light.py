# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The layers without solprop calls: messages, reduction, states, explorer."""

import numpy as np
import pytest

from solprop.grid import EnvelopeGrid
from solprop.service.messages import PROTOCOL, Event, Reply, Request
from solprop.service.reduce import Display, Reducer, bin_max
from solprop.service.states import ACTIVE, ExecutionState, advance

# ---------------------------------------------------------------------- messages


def test_messages_carry_their_identity():
    request = Request("check", {"text": "x"}, request_id=4, revision=2, key="k")
    assert request.protocol == PROTOCOL
    assert Reply("check", 4, 2, value=[]).ok
    assert not Reply("check", 4, 2, error="no").ok
    assert not Reply("check", 4, 2, superseded=True).ok
    assert all(
        Event("r", kind).terminal for kind in ("finished", "stopped", "failed", "lost")
    )


# ---------------------------------------------------------------------- reduction


def test_bin_max_keeps_each_bins_peak_and_fills_empty_bins():
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x = np.array([0.5, 0.6, 0.7, 3.5, 4.0])  # bins 1 and 2 are empty
    y = np.array([1.0, 9.0, 2.0, 4.0, 6.0])
    out = bin_max(x, y, edges)
    assert out[0] == 9.0  # the narrow peak survives
    assert out[3] == 6.0  # the upper edge belongs to the last bin
    assert out[1] == pytest.approx(np.interp(1.5, x, y))
    # Outside the samples' span an empty bin is zero.
    assert bin_max(np.array([0.1]), np.array([5.0]), edges)[2] == 0.0


def test_the_reducer_fixes_its_bins_on_the_first_plane():
    reducer = Reducer(Display(n_wavelength=4, n_time=3))
    with pytest.raises(RuntimeError, match="first plane"):
        reducer.axes()
    lam = np.array([np.inf, -1.0, 1e-6, 2e-6, 3e-6])
    density = np.array([7.0, 7.0, 1.0, 2.0, 3.0])
    t = np.linspace(-1e-13, 1e-13, 11)
    spectra = {"wavelength": (lam, density), "f": (3e8 / lam, density)}
    reduced, power = reducer.reduce(spectra, t, np.ones(11))
    assert reduced["wavelength"].dtype == np.float32
    assert reduced["wavelength"].shape == reduced["f"].shape == (4,)
    assert power.shape == (3,)
    assert float(reduced["wavelength"].max()) == 3.0  # non-physical ones left out
    axes = reducer.axes()
    assert axes["wavelength"][0] > 1e-6 and axes["time"].size == 3
    assert axes["f"][-1] < 3e14 and axes["bandpass"] is None
    # The second plane reuses the bins.
    reducer.reduce({"wavelength": (lam * 2, density)}, t * 2, np.ones(11))
    assert np.array_equal(reducer.axes()["wavelength"], axes["wavelength"])


def test_a_display_takes_the_runs_window_only_where_it_has_none():
    keywords = {"wavelength_limits": (2e-6, 2e-7)}
    assert Display().within(keywords).wavelength_range == (2e-7, 2e-6)
    fixed = Display(wavelength_range=(1e-7, 1e-6))
    assert fixed.within(keywords) is fixed
    assert Display().within({}).wavelength_range is None
    grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 200e-15)
    assert Display().within({"grid": grid}).wavelength_range == pytest.approx(
        (400e-9, 2e-6)
    )
    assert Display().within({"grid": object()}).wavelength_range is None


# ---------------------------------------------------------------------- states


def test_the_state_machine_follows_a_run():
    state = ExecutionState.QUEUED
    for event, expected in [
        ("launch", ExecutionState.STARTING),
        ("progress", ExecutionState.STARTING),
        ("started", ExecutionState.RUNNING),
        ("stop", ExecutionState.STOPPING),
        ("stopped", ExecutionState.STOPPED),
        ("started", ExecutionState.STOPPED),  # an ended run stays ended
    ]:
        state = advance(state, event)
        assert state == expected
    assert advance(ExecutionState.STOPPING, "finished") == ExecutionState.COMPLETE
    assert advance(ExecutionState.RUNNING, "lost") == ExecutionState.LOST
    assert advance(ExecutionState.STARTING, "failed") == ExecutionState.FAILED
    assert ExecutionState.COMPLETE not in ACTIVE


# ---------------------------------------------------------------------- workspace


# ---------------------------------------------------------------------- registry


# ---------------------------------------------------------------------- the explorer


def test_the_explorers_axes_and_quantities():
    from solprop.service import explore  # noqa: PLC0415 - only here

    omega = 2 * np.pi * 299_792_458.0 / np.array([1e-6, 5e-7])
    assert explore.axis_values(omega, "frequency") == pytest.approx(
        [299.792458, 599.584916]
    )
    with pytest.raises(KeyError):
        explore.axis_values(omega, "time")
    mode = {
        "neff_minus_one": np.array([1e-5, 2e-5]),
        "beta1": np.array([3.3e-9, 3.4e-9]),
        "beta2": np.array([-1e-29, 1e-29]),
        "beta3": np.array([1e-44, 1e-44]),
        "alpha": np.array([0.1, 0.2]),
        "aeff": 1e-8,
        "gamma": None,
        "phase_mismatch": None,
    }
    for quantity in explore.QUANTITIES:
        values = explore.quantity_values(mode, omega, quantity)
        if quantity.key in ("gamma", "mismatch"):
            assert values is None
        else:
            assert values is not None and values.shape == (2,)
    sets = [
        {"value": "x", "error": "no"},
        {"value": None, "omega": omega, "modes": [{"label": "HE1,1", **mode}]},
    ]
    (row,) = explore.table_rows(sets, 7e-7)
    assert row["label"] == "HE1,1" and row["gamma"] is None
    assert row["neff"] == pytest.approx(1.6e-5)
    assert explore.table_rows([{"error": "no"}], 7e-7) == []
    assert explore.table_rows(sets, 2e-6)[0]["neff"] is None  # outside the band
    assert explore.sweep_values("He, 2 bar") == ["He", "2 bar"]

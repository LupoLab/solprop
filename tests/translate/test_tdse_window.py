# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resolve the SEP 20 atomic window on the response lattice (section 7.1 item 6).

The binding's slicing, padding and causal-stopping checks are in
``tests/driver/test_tdse_hacc_binding.py``; this module covers the seam's arithmetic.
"""

import dataclasses
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tests.hacc_cases import hacc_response

from solprop.exceptions import TranslationError
from solprop.grid import FieldGrid
from solprop.nonlinear.tdse import TDSEAtomicWindow
from solprop.translate.tdse import (
    TDSEDescriptor,
    lattice_times,
    resolve_atomic_window,
)

# run_short's grid (SEP 20 section 3.1): 1024 coarse and 2048 fine samples of
# 50.03 as over 102.5 fs, with the collar flat over the requested +/- 30 fs.
_GRID = FieldGrid.from_limits(4.05, 800e-9, (9e-8, 1.6e-6), 60e-15)


def _planck(t, start, stop, taper):
    """Independent scalar Planck gate (McKechan et al. 2010, eq. 7)."""

    def edge(y):
        return 1.0 / (1.0 + math.exp(min(700.0, 1.0 / y - 1.0 / (1.0 - y))))

    if t <= start or t >= stop:
        return 0.0
    if t < start + taper:
        return edge((t - start) / taper)
    if t > stop - taper:
        return edge((stop - t) / taper)
    return 1.0


def test_run_short_grid_is_the_documented_lattice():
    assert (_GRID.n_t, _GRID.n_to) == (1024, 2048)
    assert _GRID.dt_fine == pytest.approx(50.03e-18, abs=0.01e-18)
    assert_array_equal(_GRID.t_fine, lattice_times(_GRID.n_to, _GRID.dt_fine))


@pytest.mark.parametrize("taper", [2e-15, 5e-15, 10e-15])
def test_window_bounds_samples_and_gate_match_an_independent_formula(taper):
    window = resolve_atomic_window(
        TDSEAtomicWindow(60e-15, taper), _GRID.n_to, _GRID.dt_fine
    )
    times = _GRID.t_fine
    # t_s is the first sample at or after -30 fs, t_e the last at or before +30 fs.
    first = int(np.searchsorted(times, -30e-15, side="left"))
    last = int(np.searchsorted(times, 30e-15, side="right")) - 1
    assert (window.first, window.last) == (first, last)
    assert (window.t_s, window.t_e) == (times[first], times[last])
    assert times[first - 1] < -30e-15 <= times[first]
    assert times[last] <= 30e-15 < times[last + 1]
    assert window.samples == last - first + 1
    assert window.atomic_duration == times[last] - times[first]
    # The gate is never stretched to the sample endpoints.
    assert window.gate == (-30e-15, -30e-15 + taper, 30e-15 - taper, 30e-15)
    gate = window.gate_values(times)
    expected = np.array([_planck(t, -30e-15, 30e-15, taper) for t in times])
    # Both evaluate the same smooth expression in different operation orders; the
    # exponent reaches 700 near the breakpoints, so the absolute difference is
    # bounded by about 700 eps / 4 < 1e-12.
    assert_allclose(gate, expected, rtol=0, atol=1e-12)
    flat = (times >= -30e-15 + taper) & (times <= 30e-15 - taper)
    assert_array_equal(gate[flat], 1.0)
    assert_array_equal(gate[(times <= -30e-15) | (times >= 30e-15)], 0.0)
    # Continuous at the four joins: neighbouring samples approach the join values.
    for join, value in zip(window.gate, (0.0, 1.0, 1.0, 0.0), strict=True):
        near = np.argmin(np.abs(times - join))
        assert abs(gate[near] - value) < 0.05


def test_window_on_non_aligned_bounds_and_wider_references():
    # 60.01 fs puts both bounds between samples; 100 fs is the L5 reference.
    for duration in (60.01e-15, 100e-15):
        window = resolve_atomic_window(
            TDSEAtomicWindow(duration), _GRID.n_to, _GRID.dt_fine
        )
        assert window.t_s >= -duration / 2 > window.t_s - _GRID.dt_fine
        assert window.t_e <= duration / 2 < window.t_e + _GRID.dt_fine
    narrow = TDSEDescriptor.from_response(
        hacc_response(atomic_window=TDSEAtomicWindow(60e-15)), _GRID
    )
    wide = TDSEDescriptor.from_response(
        hacc_response(atomic_window=TDSEAtomicWindow(100e-15)), _GRID
    )
    assert narrow.identity != wide.identity
    lattices = [
        {
            key: value
            for key, value in json.loads(one.configuration_json)["lattice"].items()
            if key not in ("atomic_window", "atomic_duration")
        }
        for one in (narrow, wide)
    ]
    # Only the atomic interval changes: the optical lattice and its collar do not.
    assert lattices[0] == lattices[1]
    assert (narrow.samples, narrow.spacing) == (wide.samples, wide.spacing)


def test_resolved_default_and_explicit_duration_share_one_identity():
    explicit = hacc_response(atomic_window=TDSEAtomicWindow(60e-15))
    resolved = dataclasses.replace(
        hacc_response(atomic_window="auto"),
        atomic_window=dataclasses.replace(TDSEAtomicWindow(), duration=60e-15),
    )
    assert (
        TDSEDescriptor.from_response(explicit, _GRID).identity
        == TDSEDescriptor.from_response(resolved, _GRID).identity
    )


@pytest.mark.parametrize(
    ("window", "message"),
    [
        (TDSEAtomicWindow(), "unresolved"),
        (TDSEAtomicWindow(103e-15, 5e-15), "period"),
        (TDSEAtomicWindow(2.5 * 50.0346e-18, 50e-18), "four samples"),
    ],
)
def test_window_refusals_before_any_provider(window, message):
    with pytest.raises(TranslationError, match=message):
        TDSEDescriptor.from_response(hacc_response(atomic_window=window), _GRID)


def test_full_period_window_is_admitted():
    period = _GRID.n_to * _GRID.dt_fine
    window = resolve_atomic_window(
        TDSEAtomicWindow(period, 5e-15), _GRID.n_to, _GRID.dt_fine
    )
    assert (window.first, window.last) == (0, _GRID.n_to - 1)


def test_window_requires_the_standard_fine_lattice():
    shifted = dataclasses.replace(_GRID, t_fine=_GRID.t_fine + 1e-21, t=_GRID.t + 1e-21)
    with pytest.raises(TranslationError, match="standard fine"):
        TDSEDescriptor.from_response(hacc_response(), shifted)


def test_descriptor_gate_is_read_only_and_cached_per_configuration():
    descriptor = TDSEDescriptor.from_response(hacc_response(), _GRID)
    gate = descriptor.source_gate()
    assert gate is not None and not gate.flags.writeable
    assert descriptor.source_gate() is gate
    window = descriptor.atomic_window
    assert window is not None
    assert_array_equal(gate, window.gate_values(_GRID.t_fine))

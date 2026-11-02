# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Checks of ``numba_cpu`` through the driver, moved out of the shared modules.

Each has its counterpart on the backends that stay in the module it came from. They go
with ``numba_cpu`` (SEP 43 section 6.4).
"""

import dataclasses
import json
from collections.abc import Mapping
from typing import cast

import h5py
import numpy as np
import pytest

from driver.test_coupled_input_noise import PULSE_ENERGY
from driver.test_coupled_input_noise import ROUTES as NOISE_ROUTES
from driver.test_native_resume_history import (
    assert_frozen_residual_history_refused,
    assert_same_rows_cross,
    quiet,
)
from driver.test_scans import _dataset, build_point
from solprop.backends import available
from solprop.driver.scan import Scan, run_scan

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

NUMBA_PARITY = 1e-12
"""Largest relative departure of ``numba_cpu``'s exit field from the reference's.

Dimensionless, a rounding budget fixed before the comparison: the two backends run the
same tableau on the same fixed schedule, so they differ by the rounding of ten steps of
transforms, flow actions and stage sums, each a few hundred ulps of the field's peak at
most. Measured 3.2e-15 (wall), 6.4e-16 (taper), 1.6e-16 (moving).
"""

needs_native = pytest.mark.skipif(
    "native" not in available(), reason="needs solprop-native"
)
"""The history's crossings take the native backend for their other side."""


@pytest.mark.medium
@pytest.mark.parametrize("route", list(NOISE_ROUTES))
def test_numba_cpu_carries_the_input_draw_through_a_coupled_flow_as_the_reference(
    route,
):
    """The input draw needs no backend path of its own, so ``numba_cpu`` admits it."""
    build = NOISE_ROUTES[route]
    reference = build("input", PULSE_ENERGY, "numpy_ref").run(progress="none")
    compiled = build("input", PULSE_ENERGY, "numba_cpu").run(progress="none")
    expected = np.asarray(reference.fields("fft")[-1])
    taken = np.asarray(compiled.fields("fft")[-1])
    assert np.max(abs(taken - expected)) <= NUMBA_PARITY * np.max(abs(expected))


@needs_native
@quiet
@pytest.mark.parametrize(
    ("taken", "resumed", "message"),
    [
        ("native", "numba_cpu", "native engine's applied frozen-residual"),
        ("numba_cpu", "native", "host engine's frozen-residual weights"),
    ],
)
def test_a_frozen_residual_history_does_not_cross_engines(
    build_simulation, taken, resumed, message
):
    assert_frozen_residual_history_refused(build_simulation, taken, resumed, message)


@needs_native
@quiet
@pytest.mark.parametrize("case", ["plain", "compensated"])
@pytest.mark.parametrize(
    ("taken", "resumed"), [("native", "numba_cpu"), ("numba_cpu", "native")]
)
def test_a_history_of_the_same_rows_crosses_backends(
    build_simulation, taken, resumed, case
):
    assert_same_rows_cross(build_simulation, taken, resumed, case)


def build_numba_point(point):
    simulation = build_point(point)
    simulation.settings = dataclasses.replace(simulation.settings, backend="numba_cpu")
    return simulation


def test_scan_records_numba_execution_policies_per_point(tmp_path):
    scan = Scan("numba_policies", power=[1.0, 2.0])
    destination = tmp_path / "policies.h5"
    result = run_scan(
        scan,
        build_numba_point,
        output=destination,
        fields="none",
        stats="none",
        progress="none",
    )
    assert result.status.tolist() == ["done", "done"]
    keys = {
        "parallel_nodes",
        "kernel_regime_elements",
        "modal_projection",
        "raman_fftw_threads",
        "raman_tail_nodes",
        "raman_tail_fftw_threads",
        "plasma_fraction_reuse",
        "column_plasma_magnitude",
    }
    with h5py.File(destination, "r") as file:
        for index in range(2):
            backend = json.loads(str(_dataset(file, "points/backend").asstr()[index]))
            assert backend["name"] == "numba_cpu"
            settings = backend["settings"]
            assert keys <= settings.keys()
            metadata = cast(Mapping, result.point(index)["metadata"])
            assert metadata["backend"]["settings"] == settings

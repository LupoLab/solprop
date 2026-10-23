# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A tapered capillary beside a steep pressure gradient translates and runs.

The logarithmic power weight on the two-parameter (density x radius) box would fail
in translation for a steep gradient beside any taper if its tolerance came from the
round-off floor measured along the path, since the parameter search meets the floor
on the whole box, whose low-density, wide-bore corner the path never visits. These are
the reproducing cases, on their own grid.
"""

import functools
import warnings
from dataclasses import replace

import numpy as np
import pytest

from solprop.boundaries import absorber
from solprop.driver.simulation import Simulation
from solprop.exceptions import (
    GasFlowWarning,
    MaterialRangeWarning,
)
from solprop.profiles import Linear
from solprop.stepper.schedule import FixedSchedule

LENGTH = 0.1
"""Capillary length, m."""

TAPER = Linear(100e-6, 80e-6, LENGTH)
"""A 20 % taper, m."""

SLIGHT = Linear(100e-6, 99e-6, LENGTH)
"""A 1 % taper, m."""

STEPS = 10

QUIET = (MaterialRangeWarning, GasFlowWarning)
"""What the reproducing cases warn about on purpose: the argon index is extrapolated at
the band's short end, and a 5 bar difference across a 100 um bore is not laminar (Re
about 8000), which is the physics and not the translation under test.
"""


def _build(radius, pressure, vacuum_end, **overrides):
    """Build the reproducer: argon, 30 fs, 10 uJ at 800 nm."""
    extra = {} if vacuum_end is None else {"vacuum_end": vacuum_end}
    with warnings.catch_warnings():
        for category in QUIET:
            warnings.simplefilter("ignore", category)
        return Simulation.from_capillary(
            radius,
            LENGTH,
            "Ar",
            pressure,
            lambda0=800e-9,
            fwhm=30e-15,
            energy=10e-6,
            wavelength_limits=(300e-9, 2e-6),
            trange=400e-15,
            **extra,
            **overrides,
        )


@pytest.mark.medium
@pytest.mark.parametrize(
    ("radius", "pressure", "vacuum_end"),
    [
        # Failed before the fix.
        pytest.param(TAPER, (5.0, 0.2), "floor", id="taper-5-0.2"),
        pytest.param(SLIGHT, (5.0, 0.2), "floor", id="slight-5-0.2"),
        pytest.param(TAPER, (5.0, 0.0), "floor", id="taper-5-0-floor"),
        pytest.param(TAPER, (0.0, 5.0), "floor", id="taper-0-5-floor"),
        # Built before the fix, and bitwise the same after it.
        pytest.param(TAPER, (5.0, 0.0), None, id="taper-5-0-choked"),
        pytest.param(TAPER, (5.0, 0.5), "floor", id="taper-5-0.5"),
        pytest.param(TAPER, (2.0, 0.2), "floor", id="taper-2-0.2"),
        pytest.param(TAPER, (1.0, 0.05), "floor", id="taper-1-0.05"),
        pytest.param(100e-6, (5.0, 0.2), "floor", id="uniform-5-0.2"),
    ],
)
def test_a_tapered_gradient_translates_on_the_parameter_expansion(
    radius, pressure, vacuum_end
):
    with warnings.catch_warnings():
        for category in QUIET:
            warnings.simplefilter("ignore", category)
        problem = _build(radius, pressure, vacuum_end).translate()
    # The expansion carries the logarithmic term itself: no z-table fallback, which
    # Kokkos would refuse beside the parameter expansion.
    assert problem.linear.phi_representation == "expansion"
    expansion = problem.linear.phi_expansion
    assert expansion is not None and expansion.endpoint_degrees is not None


@functools.cache
def _run(backend):
    simulation = _build(
        TAPER,
        (5.0, 0.2),
        "floor",
        plasma=False,
        n_save=3,
        schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        backend=backend,
    )
    # A collar referred to ten metres, weak enough for ten explicit centimetre steps
    # (ADR-025); the default refers it to the run and needs more of them.
    simulation.settings = replace(
        simulation.settings, absorber=absorber(simulation.grid, boundary_length=10.0)
    )
    with warnings.catch_warnings():
        for category in QUIET:
            warnings.simplefilter("ignore", category)
        return simulation.run()


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_a_formerly_failing_tapered_gradient_runs_ten_steps_on_every_backend(backend):
    if backend == "native":
        pytest.importorskip("solprop_native")
    field = np.asarray(_run(backend)["field"])
    assert np.all(np.isfinite(field))
    reference = np.asarray(_run("numpy_ref")["field"])
    scale = float(np.linalg.norm(reference[-1]))
    # The same ten fixed steps on every backend: what differs is the order of the
    # floating-point sums, a few ulps per step compounded over ten (measured 2.7e-16
    # on numba_cpu and 3.9e-16 on the native backend).
    assert np.linalg.norm(field[-1] - reference[-1]) / scale < 1e-10

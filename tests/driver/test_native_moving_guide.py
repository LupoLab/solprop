# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A moving solid guide on the native CPU backend, against ``numpy_ref``.

Native executes a moving guide's synthesis and projection maps (kernel schema 14): the
maps' Chebyshev coefficients along z are uploaded once at construction and combined per
stage from the bundle's ``transform_weights`` by a native kernel -- the weighted sum
over the stage's section's ``n_deg`` terms, one team per map row and the row's bins its
vector lanes -- which feeds the general modal synthesis, contraction and projection
unchanged. The connection rides in the frozen-residual coupled flow exactly as the
capillary flows execute it.

The cases are those of ``tests/driver/test_moving_guide.py`` (``moving_guide_cases``):
the kinked graded-index taper, the step-index HE11/HE12 taper, and the same taper in a
complex basis, where the connection's conjugation matters (a circular HE11 pair beside
the parity pair of HE12). Their ten fixed steps against ``numpy_ref``, within the bound
written before the run (``moving_guide_cases.moving_parity_bound``), are rows of the
case catalogue (``tests/backends/flow_rows.py``, ``moving guide ...``), guarded by the
maps frozen at the launch. What stays here: the reference's own witness that a stage's
maps are not the step's first, the entry point's ``auto`` run and the thread count.
"""

import dataclasses
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from coupled_flow_cases import (
    propagate,
    reference_and_linear,
)
from driver.test_native_coupled_flow import thread_states
from modal_provider_cases import grin_grid
from moving_guide_cases import (
    CASES,
    GRIN_PULSE,
    LENGTH,
    STEPS,
    grin_kinked,
    grin_taper,
    kinked_radius,
    moving_parity_bound,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.output.memory import MemoryOutput

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)


@pytest.mark.medium
def test_the_moving_maps_are_combined_per_stage_not_frozen():
    """The stages of one step read different maps: a frozen map would not be caught.

    The reference run with every stage's weights replaced by the step's first
    stage's -- the maps frozen at the step's start -- departs from ``numpy_ref``'s own
    run by far more than the parity bound, so the parity test above measures the
    combination and not only the rest of the route. The stages are the Runge-Kutta
    march's: an Adams step evaluates the source once, at the step's end. The frozen
    maps are formed on the reference's host-formed bundles; the native trial reads its
    stage tables natively and has no bundle to freeze.
    """
    built = grin_kinked()
    stepper = dataclasses.replace(built.run_request.stepper, integrator=None)
    case = dataclasses.replace(
        built, run_request=built.run_request.with_stepper(stepper)
    )
    expected = propagate(case, NumpyRefBackend)
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    bound = moving_parity_bound(case, expected, linear)
    backend = NumpyRefBackend(case.problem, case.run_request, grid=case.grid)
    single = backend._stage_scalars

    def frozen(z0, z1):
        first = single(z0, z0).transform_weights
        return dataclasses.replace(single(z0, z1), transform_weights=first)

    # The reference's trial reads each stage's bundle through `_stage_scalars`.
    backend._stage_scalars = frozen
    try:
        backend.set_initial(case.problem.initial.field0, None)
        backend.propagate_to(case.length)
        actual = backend.observe().require_field().copy()
    finally:
        backend.close()
    assert float(np.max(np.abs(actual - expected))) > 100 * bound


@pytest.mark.medium
def test_auto_runs_a_moving_guide_on_native():
    """``backend="auto"`` resolves to Kokkos, which now executes the moving maps."""
    guide = grin_taper(kinked_radius(), kerr=True)
    grid = grin_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = Simulation.from_waveguide(
            (),
            LENGTH,
            provider=guide,
            grid=grid,
            save_planes=[LENGTH],
            dz_max=LENGTH / STEPS,
            dz_init=LENGTH / STEPS,
            rtol=1e-8,
            backend="auto",
            **{"noise": None, **GRIN_PULSE},
        )
        # The bare guide, no absorber and so no window watch: the light reaches the
        # edge of this short window by design.
        simulation.settings = dataclasses.replace(
            simulation.settings, absorber=None, boundary_check=None
        )
        run = cast(MemoryOutput, simulation.run(progress="none"))
    assert run.metadata is not None
    record = run.metadata.backend
    assert record is not None
    assert record.name == "native"
    energy = [float(np.sum(plane.stats["physical_energy"])) for plane in run.planes()]
    assert len(energy) == 2
    # As the default's end-to-end run on the reference: lossless, so the run's
    # physical energy moves by its integration error alone, 5.2e-11 there.
    assert abs(energy[-1] / energy[0] - 1.0) < 1e-9


PROBE = Path(__file__).with_name("native_moving_thread_probe.py")
"""The subprocess that runs the cases at one thread count and saves their states."""


@pytest.mark.slow
def test_a_moving_guide_is_invariant_under_the_thread_count(tmp_path):
    """One thread against four on the three cases, within the parity bound.

    Every pool is forced (:func:`~driver.test_native_coupled_flow.thread_states`). The
    combination is a fixed-order loop inside one output element, as every other kernel
    of the route is. Bitwise invariance is not promised, though (SEP 25): two thread
    layouts are two orderings of one discrete march, held to the bound of two
    implementations of it, :func:`moving_parity_bound`.
    """
    names = sorted(CASES)
    one = thread_states(PROBE, 1, names, tmp_path)
    four = thread_states(PROBE, 4, names, tmp_path)
    for name in names:
        case = CASES[name]()
        bound = moving_parity_bound(case, *reference_and_linear(case))
        assert np.any(one[name])
        departure = float(np.max(np.abs(four[name] - one[name])))
        assert departure <= bound, (name, departure, bound)

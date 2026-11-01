# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preserve labelled tensor configuration and existing noise through HDF5 resume."""

import numpy as np
import pytest

from driver.test_resume import read_file, run_until
from march_bounds import (
    FIXED_ORDER,
    STATISTIC_FACTOR,
    assert_matched,
    assert_points_matched,
    simulation_bound,
)
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.executors import Batch, Local
from solprop.driver.scan import Scan, run_scan
from solprop.fourier import to_internal
from solprop.noise import input_noise, modified_noise
from solprop.output.hdf5 import resume
from solprop.polarisation import horizontal, rotate, waveplate
from solprop.stats import StatsSpec
from vector_raman_cases import raman_response, vector_problem


@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
@pytest.mark.parametrize("noise_model", ["input", "modified"])
def test_tensor_run_resumes_with_the_same_realisation_and_configuration(
    backend, noise_model, tmp_path, record_property
):
    if backend == "native":
        pytest.importorskip("solprop_native")
    problem, grid, modes = vector_problem(steps=8)
    make_noise = input_noise if noise_model == "input" else modified_noise
    draw = make_noise(grid, np.random.default_rng(4040), n_modes=1, n_pol=2)

    def simulation():
        return Simulation(
            grid,
            modes,
            [raman_response("H2", thg=False)],
            problem.initial.field0[0],
            noise=draw,
            settings=SimulationSettings(
                backend=backend,
                density=1e25,
                save_planes=[grid.length / 2, grid.length],
                schedule=problem.stepper.schedule,
                # No order choice on the prescribed schedule: the resumed run and
                # the uninterrupted one are one discrete march.
                integrator=FIXED_ORDER,
                dz_init=grid.length / 8,
                dz_max=grid.length / 8,
                controller=problem.stepper.controller,
                stats=StatsSpec(names=("energy",)),
            ),
        )

    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    simulation().run(whole, progress="none")
    run_until(simulation(), part, after=1)
    simulation().run(part, resume=True, progress="none")
    march = simulation_bound(simulation(), backend=backend)
    for key in ("field", "z", "stats/energy", "noise_phase"):
        expected, actual = read_file(whole, key), read_file(part, key)
        if key in ("field", "stats/energy"):
            scale = float(np.max(np.abs(expected)))
            shift = float(np.max(np.abs(actual - expected))) / scale
            record_property(f"power_resume_{key.replace('/', '_')}", shift)
            # Not bitwise: the sqrt(w) output/resume map rounds twice, and the two
            # runs are two executions of one march (SEP 25). The field keeps the
            # prescribed 2.7e-16 global-amplitude bound plus the march's normwise
            # budget, which bounds every element relative to the peak too; energy
            # adds the spectral sum and one squared-modulus operation. Measured
            # maxima: field 1.77e-16; energy 2.64e-16.
            if key == "field":
                bound = 2.7e-16 + march
            else:
                bound = 8 * np.finfo(float).eps + STATISTIC_FACTOR * march
            np.testing.assert_allclose(actual, expected, rtol=0, atol=bound * scale)
        elif key == "z":
            # The prescribed save positions: exact.
            np.testing.assert_array_equal(actual, expected)
        else:
            # The same stored realisation carried by one march: its rounding budget.
            assert_matched(actual, expected, march, what=key)
    metadata = resume(part).metadata
    assert metadata.schema_version == 2
    record = repr(metadata.configuration)
    assert "tensor_terms" in record and "rank2" in record and "isotropic" in record


# A native lockstep batch replays its Adams members on solprop's march, with the
# warning native/tests/test_batch.py asserts (SEP 39).
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.medium
@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_jones_tensor_scan_batches_match_singletons_with_a_partial_group(
    tmp_path, backend
):
    if backend == "native":
        pytest.importorskip("solprop_native")
    problem, grid, modes = vector_problem(steps=2)

    def build(point):
        jones = to_internal(
            rotate(waveplate(np.pi / 2), float(point.values["angle"])) @ horizontal()
        )
        initial = problem.initial.field0[0, :, :1] * jones[None, :, None]
        return Simulation(
            grid,
            modes,
            [raman_response("H2", thg=False)],
            initial,
            settings=SimulationSettings(
                backend=backend,
                density=1e25,
                save_planes=[grid.length],
                schedule=problem.stepper.schedule,
                integrator=FIXED_ORDER,
                dz_init=grid.length / 2,
                dz_max=grid.length / 2,
                controller=problem.stepper.controller,
                stats=StatsSpec(names=("energy",)),
            ),
        )

    scan = Scan("vector Raman Jones scan", angle=[0.0, np.pi / 8, np.pi / 4])
    outputs = [
        run_scan(
            scan,
            build,
            output=tmp_path / f"{name}.h5",
            executor=executor,
            fields="planes",
            field_planes=[-1],
            progress="none",
        )
        for name, executor in (("local", Local()), ("batch", Batch(2)))
    ]
    # A batch member is its singleton's march: within its rounding budget.
    assert_points_matched(outputs[1], outputs[0], scan, build, what="batch member")

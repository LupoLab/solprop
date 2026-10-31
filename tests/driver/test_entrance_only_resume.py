# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resume legacy entrance-only files without treating the launch as interpolated."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from solprop.driver import Simulation
from solprop.output import MemoryOutput, open_file, resume
from solprop.output.schema import INTERPOLATED_RESUMES_ANNOTATION
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep


def test_entrance_without_terminal_checkpoint_resumes_dense_run_normally(tmp_path):
    step = 2.0**-20
    simulation = Simulation.from_gnlse(
        0.0,
        4 * step,
        [0.0, 0.0, -1e-26],
        wavelength=800e-9,
        wavelength_limits=(500e-9, 1.6e-6),
        trange=200e-15,
        fwhm=20e-15,
        peak_power=1.0,
        save_planes=[1.5 * step, 4 * step],
        backend="numpy_ref",
        integrator=Multistep(),
        dz_init=step,
        dz_max=step,
        noise=None,
        interpolate_planes=True,
        stats=StatsSpec(names=("energy",)),
    )
    reference = simulation.run(progress="none")
    assert isinstance(reference, MemoryOutput)
    assert reference.metadata is not None
    entrance_only = MemoryOutput()
    entrance_only.open(reference.metadata)
    entrance_only.save(reference.planes()[0])
    path = tmp_path / "entrance-only.h5"
    # Old files have regular planes but no explicit terminal checkpoint. Keep the
    # current scientific identity: this is not a cross-schema resume conversion.
    entrance_only.to_file(path)
    state = resume(path)
    assert state.n_saved == 1
    assert state.plane.z == 0.0
    assert state.history.n_steps == 0
    assert state.checkpoint is None
    assert state.outcome is None

    simulation.run(path, resume=True, progress="none")
    with open_file(path) as resumed:
        assert INTERPOLATED_RESUMES_ANNOTATION not in resumed.annotations
        assert_array_equal(resumed["z"], reference["z"])
        expected = reference.fields(convention="fft")
        # The resumed initial state only crosses the output convention boundary;
        # permit roundoff in that conversion and the four linear propagators.
        assert_allclose(
            resumed.fields(convention="fft"),
            expected,
            rtol=0.0,
            atol=64 * np.finfo(float).eps * np.max(np.abs(expected)),
        )

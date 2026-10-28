# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 14 phase 3: saving the march's own steps beside the requested planes.

``step_planes=k`` stops the march at the end of every ``k``-th accepted step and saves
it: no extra step and no interpolation, so every plane is a state the march passed
through and a run stopped on one resumes as from any landed plane.

The runs are adaptive, so a run with saved steps is not promised the plain run's steps
or bits (SEP 25 section 4.4). Within one run the layout is exact: its saved positions
are its own accepted step ends and the requested planes. Across two runs the work is a
band and the fields an accuracy envelope against the same soliton at a hundredth of
the tolerance (``tests/march_bounds.py``).
"""

import numpy as np
import pytest

from bounds import assert_work_within
from march_bounds import TIGHTER, assert_in_envelope
from solprop.backends import available
from solprop.driver import Simulation
from solprop.driver.simulation import _interpolated
from solprop.exceptions import TranslationError
from solprop.log import NullReporter
from solprop.output.hdf5 import resume
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.stepper.multistep import Multistep

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

BACKENDS = ["numpy_ref", "numba_cpu"] + (["native"] if "native" in available() else [])


def soliton(**extra):
    """Return a second-order soliton over a quarter of its period, three planes."""
    return Simulation.from_gnlse(
        0.11,
        0.012,
        [0.0, 0.0, -1.183e-26],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=4 * 9.8,
        pulseshape="sech",
        n_save=4,
        noise=None,
        **{"integrator": Multistep(), **({"rtol": RTOL, **extra})},
    )


def accepted_ends(output):
    """Return the end of every accepted step of a run, m."""
    ends = []
    for plane in output.planes():
        history = plane.step_history
        ends += list((history.z + history.h)[history.accepted])
    return np.array(ends)


RTOL = 1e-6
"""The soliton runs' relative tolerance."""

WORK_BAND = 1.1
"""How far apart two adaptive runs' accepted step counts may be: a fork's few steps."""


def tight(backend):
    """Return the soliton's requested planes at a hundredth of the tolerance.

    On the default method whatever the run compared with uses: a reference is the same
    problem solved more accurately, and the Runge-Kutta march here is held by its step
    growth limit rather than its tolerance, so tightening it alone changes nothing.
    """
    run = soliton(rtol=RTOL / TIGHTER, backend=backend).run(progress="none")
    return run.fields(convention="fft")[1:]


def requested(output, planes):
    """Return the fields a run saved at the requested planes."""
    z = np.asarray(output["z"])
    fields = output.fields(convention="fft")
    return np.array([fields[int(np.argmin(np.abs(z - plane)))] for plane in planes])


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_step_is_saved_at_no_extra_step(backend):
    plain = soliton(backend=backend).run(progress="none")
    saved = soliton(backend=backend, step_planes=1).run(progress="none")
    ends = accepted_ends(saved)
    # No extra step: the work of the plain march, within an adaptive pair's band.
    assert_work_within(ends.size, accepted_ends(plain).size, factor=WORK_BAND)
    # Every saved plane is one of the run's own accepted step ends, exactly, and the
    # requested planes are landed on exactly (ADR-009).
    z = np.asarray(saved["z"])
    np.testing.assert_array_equal(z[1:], ends)
    planes = np.asarray(plain["z"])[1:]
    assert set(planes) <= set(z)
    assert_in_envelope(
        [requested(saved, planes)],
        tight(backend),
        plain.fields(convention="fft")[1:],
        what="the requested planes",
    )
    assert saved.metadata is not None
    assert saved.metadata.stepper.limits.step_planes == 1


def test_thinning_saves_every_kth_step_and_the_requested_planes():
    thinned = soliton(step_planes=3).run(progress="none")
    # Within the run: every third of its own accepted step ends, and the requested
    # planes, which are landed on exactly.
    ends = accepted_ends(thinned)
    planes = np.asarray(soliton().settings.save_planes)
    expected = np.union1d(ends[2::3], planes)
    np.testing.assert_array_equal(np.asarray(thinned["z"])[1:], expected)


@pytest.mark.parametrize(
    "dense",
    [
        {"integrator": RungeKutta(), "interpolate_planes": True},
        {"integrator": Multistep(), "interpolate_planes": True},
    ],
    ids=["runge-kutta", "adams"],
)
def test_saved_steps_combine_with_dense_observation(dense):
    alone = soliton(backend="numpy_ref", **dense).run(progress="none")
    both = soliton(backend="numpy_ref", step_planes=2, **dense).run(progress="none")
    ends = accepted_ends(both)
    assert_work_within(ends.size, accepted_ends(alone).size, factor=WORK_BAND)
    z = np.asarray(both["z"])
    assert np.all(np.diff(z) > 0)
    planes = np.asarray(alone["z"])[1:]
    np.testing.assert_array_equal(np.union1d(ends[1::2], planes), z[1:])
    assert_in_envelope(
        [requested(both, planes)],
        tight("numpy_ref"),
        alone.fields(convention="fft")[1:],
        what="the interpolated planes",
    )


def test_adams_saves_its_steps_at_no_extra_call():
    method = {"integrator": Multistep(), "backend": "numpy_ref"}
    plain = soliton(**method).run(progress="none")
    saved = soliton(step_planes=1, **method).run(progress="none")
    assert_work_within(
        accepted_ends(saved).size, accepted_ends(plain).size, factor=WORK_BAND
    )
    np.testing.assert_array_equal(np.asarray(saved["z"])[1:], accepted_ends(saved))


class Stop(NullReporter):
    """A progress reporter that stops the run after a number of saved planes."""

    def __init__(self, after):
        self.after = after
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen >= self.after:
            raise RuntimeError("stopped")


def test_a_run_stopped_on_a_saved_step_resumes(tmp_path):
    path = tmp_path / "steps.h5"
    whole = soliton(step_planes=1).run(progress="none")
    with pytest.raises(RuntimeError, match="stopped"):
        soliton(step_planes=1).run(path, progress=Stop(5))
    state = resume(path)
    assert not _interpolated(state)
    soliton(step_planes=1).run(path, resume=True, progress="none")
    finished = resume(path)
    assert finished.plane.z == pytest.approx(0.012, rel=1e-12)
    # A fresh start of the controller: the same requested planes to the tolerance.
    final = np.asarray(whole.fields(convention="fft")[-1])
    error = np.max(np.abs(finished.plane.field - final)) / np.max(np.abs(final))
    assert error < 1e-5


def test_the_record_keys_on_the_thinning():
    assert soliton(step_planes=2).metadata().configuration_hash() != (
        soliton().metadata().configuration_hash()
    )


def test_a_negative_count_is_refused():
    with pytest.raises(TranslationError, match="step_planes must be 0"):
        soliton(step_planes=-1).translate()


def test_a_scan_refuses_saved_steps(tmp_path):
    from solprop.driver.scan import Scan, run_scan  # noqa: PLC0415 - the scan machinery

    result = run_scan(
        Scan("scan", power=[1.0]),
        lambda point: soliton(step_planes=1, backend="numpy_ref"),
        output=tmp_path / "scan.h5",
        progress="none",
    )
    assert result.status.tolist() == ["failed"]
    assert "step_planes" in str(result.point(0)["error"])


@pytest.mark.parametrize(
    "dense",
    [
        {"integrator": RungeKutta(), "interpolate_planes": True},
        {"integrator": Multistep(), "interpolate_planes": True},
    ],
    ids=["runge-kutta", "adams"],
)
def test_a_step_that_crosses_an_interpolated_plane_is_saved_after_it(dense):
    # With every step saved, the step that crosses a requested plane is saved too, in
    # order after the plane interpolated inside it.
    alone = soliton(backend="numpy_ref", **dense).run(progress="none")
    every = soliton(backend="numpy_ref", step_planes=1, **dense).run(progress="none")
    z = np.asarray(every["z"])
    assert np.all(np.diff(z) > 0)
    np.testing.assert_array_equal(
        z[1:], np.union1d(accepted_ends(every), np.asarray(alone["z"])[1:])
    )


@pytest.mark.parametrize(
    "integrator", [Multistep(), RungeKutta()], ids=["default", "runge_kutta"]
)
def test_a_replayed_schedule_saves_its_steps(integrator):
    # Each integrator replays a schedule in a loop of its own.
    schedule = FixedSchedule.uniform(
        0.0, 0.012, 0.001, save_planes=[0.004, 0.008, 0.012]
    )
    # Without the collar, which refuses twelve fixed steps; the planes are the point.
    replay = soliton(
        backend="numpy_ref",
        schedule=schedule,
        step_planes=1,
        integrator=integrator,
        absorber=False,
    )
    z = np.asarray(replay.run(progress="none")["z"])
    np.testing.assert_allclose(z, np.linspace(0.0, 0.012, 13), rtol=0, atol=1e-15)

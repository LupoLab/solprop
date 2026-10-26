# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A coupled-flow run and a moving-guide run resume onto the uninterrupted march.

On the host engines, which keep the Adams march's state in the file (ADR-046), a resume
continues the same discrete march: the same fixed schedule, the same fixed-order Adams
method and the same history, so the resumed run differs from the uninterrupted one by
rounding alone (SEP 25 section 4.4, class R). Kokkos starts the history afresh on a
resume, and its rows are bounded by the step error that costs instead.

The flow's key and its refusals are tested elsewhere; here a run carrying
``linear.coupled_flow`` is stopped partway and continued. A constant scalar wall, a
varying (tapered) wall and the moving graded-index guide each run into an HDF5 file
uninterrupted, and again stopped after their first saved plane and resumed from the file
by a fresh simulation object, as a new process would.

**The moving guide's resumed state is the stopped one.** A replayed schedule takes the
same steps whatever it is resumed from; the state's round trip through the file is a
conjugation and a sign flip on a route whose power weight is one; and the flow holds
nothing across a plane that is not rebuilt from the problem -- its actions are
functions of the section and the step length alone, and the stage bundle of the anchor
``z``. So what remains is the two runs' own rounding, ``tests/flow_parity.py``'s
ten-step bound (``tests/driver/test_resume.py`` is the model).

**The capillary wall's is within ``2 u`` of it.** Its file stores the true-power
amplitude ``b = sqrt(w) a`` with ``w = Re n_eff`` of each channel, and a resume divides
by the same ``sqrt(w)``: two correctly rounded operations, carried to the end by the
flow's amplification, :func:`resumed_bound`. Exact still is everything the replayed
schedule prescribes -- the planes' positions, the step history's positions, lengths and
acceptances -- and the integer accounting; the planes written before the stop, by the
interrupted run, are a second run of the same march, within the same bound.

Ten steps over a fifth of a millimetre for the walls, ten over a millimetre for the
moving guide.
"""

import dataclasses
import math

import h5py
import numpy as np
import pytest

from driver.test_coherent_capillary import LENGTH, STEPS, capillary, varying
from driver.test_moving_guide import GRIN_PULSE, grin_taper, kinked_radius
from driver.test_resume import run_until
from flow_parity import UNIT, departure, flow_scales, ten_step_bound
from march_bounds import FIXED_ORDER, STATISTIC_FACTOR, assert_stored_matched
from modal_provider_cases import grin_grid
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.exceptions import GrazingIncidenceWarning, MaterialRangeWarning
from solprop.modes import CoherentWall
from solprop.output.hdf5 import resume
from solprop.stepper import FixedSchedule

pytestmark = [
    pytest.mark.filterwarnings(
        f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
    ),
    pytest.mark.filterwarnings(
        f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
    ),
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]
"""The wall cases' deliberate warnings (``tests/driver/test_coherent_capillary.py``)."""

UNCOMPARED = {
    "meta/json",
    "meta/outcome",
    "meta/annotations/run_summary",
    "step_history/wall_time",
}
"""Metadata and outcome are checked separately; the wall clock is not reproducible,
and a resumed file's start-of-run summaries record its two starts."""


def stored(path) -> dict[str, np.ndarray]:
    """Return every dataset of a closed output file but :data:`UNCOMPARED`."""
    arrays = {}

    def visit(name, entry):
        if isinstance(entry, h5py.Dataset) and name not in UNCOMPARED:
            arrays[name] = np.asarray(entry[()])

    with h5py.File(path, "r") as file:
        file.visititems(visit)
    return arrays


def wall_case(kind, backend):
    """Return a ten-step wall run saving the entrance, the middle and the end."""
    planes = [0.0, LENGTH / 2, LENGTH]
    arguments = {
        "backend": backend,
        "n_save": 3,
        "schedule": FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / STEPS, save_planes=planes
        ),
        "integrator": FIXED_ORDER,
    }
    if kind == "constant":
        return capillary(wall=CoherentWall(), **arguments)
    if kind == "lossless":
        # The taper coupling without a wall.
        return varying(loss=False, cladding=1.45, **arguments)
    return varying(wall=CoherentWall(), **arguments)


@pytest.fixture(scope="module")
def kinked():
    """The kinked graded-index taper with a Kerr core, solved once for the module."""
    return grin_taper(kinked_radius(), kerr=True)


def moving_case(guide, backend):
    """Return a ten-step run of the kinked graded-index taper, two planes."""
    length = 1e-3
    planes = [length / 2, length]
    simulation = Simulation.from_waveguide(
        (),
        length,
        provider=guide,
        grid=grin_grid(),
        save_planes=planes,
        dz_max=length / STEPS,
        dz_init=length / STEPS,
        schedule=FixedSchedule.uniform(0.0, length, length / STEPS, save_planes=planes),
        integrator=FIXED_ORDER,
        backend=backend,
        **GRIN_PULSE,
    )
    # No collar, as the moving guide's own tests run it.
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    return simulation


def backends(*kinds):
    """Return (kind, backend) parameters: the constant wall medium, the rest slow."""
    return [
        pytest.param(
            kind,
            name,
            marks=[pytest.mark.medium if kind == "constant" else pytest.mark.slow]
            + (
                []
                if name in available()
                else [pytest.mark.skip(reason=f"{name} is not installed")]
            ),
            id=f"{kind}-{name}",
        )
        for kind in kinds
        for name in ("numpy_ref", "numba_cpu", "native")
    ]


def interrupted(build, tmp_path, *, planes):
    """Run whole, run stopped after the first plane, resume; return both files."""
    whole, part = tmp_path / "whole.h5", tmp_path / "part.h5"
    build().run(whole, progress="none")
    run_until(build(), part, after=1)
    # Stopped at the plane after the entrance: the rest comes from the resume.
    assert stored(part)["z"].size == planes - 1
    build().run(part, resume=True, progress="none")
    whole_state, resumed_state = resume(whole), resume(part)
    first, second = whole_state.metadata, resumed_state.metadata
    whole_outcome, resumed_outcome = whole_state.outcome, resumed_state.outcome
    assert whole_outcome is not None and resumed_outcome is not None
    assert whole_outcome.status == resumed_outcome.status == "complete"
    assert whole_outcome.z == resumed_outcome.z == stored(whole)["z"][-1]
    assert whole_outcome.previous_executions == ()
    assert len(resumed_outcome.previous_executions) == 1
    interruption = resumed_outcome.previous_executions[0]
    assert interruption["status"] == "failed"
    assert interruption["z"] == stored(whole)["z"][planes - 2]
    assert isinstance(interruption["message"], str)
    assert "stopped after plane 1" in interruption["message"]
    assert first.configuration_hash() == second.configuration_hash()
    assert first.modal is not None and "flow" in first.modal
    assert first.modal == second.modal
    expected, got = stored(whole), stored(part)
    assert expected["z"].size == planes
    assert set(expected) == set(got)
    return expected, got


SCHEDULED = ("z", "step_history/z", "step_history/h", "step_history/accepted")
"""What a replayed schedule fixes, whatever the state it carries: exact."""


def continued(arrays) -> bool:
    """Return whether a file kept the Adams march's state, so a resume continued it."""
    return any(name.startswith("march_snapshot/") for name in arrays)


def resumed_bound(simulation, final) -> float:
    """Return the bound on a resumed wall run's last plane, relative.

    **Stated before the measurement.** The resumed state is the stopped one times
    ``(1 + d1)(1 + d2)``, one rounding at the save and one at the resume, so within
    ``2 u`` of it relatively (the ``sqrt(w)`` is the same value both times). From
    there the two runs do the same arithmetic on inputs that differ by that much:
    the departure is the launch perturbation carried by the ten-step amplification
    ``C^S exp(Phi) (||a_0|| / ||a_L||)`` of ``tests/flow_parity.py``, plus the two
    runs' own roundings on their two trajectories, which that module's ``B_10``
    bounds (it bounds two backends' different arithmetic over ten steps; the same
    arithmetic over the five steps left is less). Both are taken over the whole run.
    """
    problem = simulation.translate()
    scales = flow_scales(problem, simulation.grid, final)
    carried = scales.condition**STEPS * math.exp(scales.phi) * scales.attenuation
    return 2.0 * UNIT * carried + ten_step_bound(scales)


@pytest.mark.parametrize(
    ("kind", "backend"), backends("constant", "varying", "lossless")
)
def test_a_wall_run_stopped_and_resumed_is_the_uninterrupted_run(
    kind, backend, tmp_path, record_property
):
    """A scalar wall, constant or tapered, continued from its file.

    Exact: the schedule's arrays. Within :func:`resumed_bound`: every plane written
    before the stop (by the interrupted run, the same march), the last plane's field,
    and its energy statistic.
    """
    expected, got = interrupted(lambda: wall_case(kind, backend), tmp_path, planes=3)
    bound = resumed_bound(wall_case(kind, "numpy_ref"), expected["field"][-1])
    for name in SCHEDULED:
        assert np.array_equal(expected[name], got[name]), name
    assert_stored_matched(
        {name: value[:2] for name, value in expected.items() if planned(value)},
        {name: value[:2] for name, value in got.items() if planned(value)},
        bound,
        what="before the stop",
    )
    measured = departure(got["field"][-1], expected["field"][-1])
    record_property("bound", bound)
    record_property("measured", measured)
    # Not vacuous: far below the 1e-9 the wall tests resolve the physics to.
    assert 0.0 < bound < 1e-10
    if not continued(got):
        # An engine that starts the Adams history afresh (Kokkos) restarts at order
        # one for the five coarse steps left: a step error, measured 6.0e-4.
        bound = 2e-3
    assert measured <= bound
    energies = [float(np.sum(one["stats/energy"][-1])) for one in (expected, got)]
    assert energies[1] == pytest.approx(
        energies[0], rel=STATISTIC_FACTOR * bound, abs=0.0
    )


def planned(value) -> bool:
    """Return whether a dataset holds one row per saved plane (three here)."""
    return bool(value.ndim) and value.shape[0] == 3


@pytest.mark.parametrize(("kind", "backend"), backends("moving"))
def test_a_moving_guide_run_stopped_and_resumed_is_the_uninterrupted_run(
    kinked, kind, backend, tmp_path
):
    """The kinked graded-index taper continued from its file, to rounding."""
    assert kind == "moving"
    expected, got = interrupted(
        lambda: moving_case(kinked, backend), tmp_path, planes=3
    )
    # Two runs of the same march: the flow's ten-step rounding bound, nothing to carry
    # from the resume (its round trip is a conjugation and a sign flip).
    simulation = moving_case(kinked, "numpy_ref")
    bound = ten_step_bound(
        flow_scales(simulation.translate(), simulation.grid, expected["field"][-1])
    )
    if not continued(got):
        # An engine that starts the Adams history afresh (Kokkos): what the schedule
        # fixes stays exact, the planes before the stop are a second run of the same
        # march, and the last plane is a step error away, the history restarting at
        # order one (measured 2.5e-4).
        for name in SCHEDULED:
            assert np.array_equal(expected[name], got[name]), name
        assert departure(got["field"][:2], expected["field"][:2]) <= bound
        measured = departure(got["field"][-1], expected["field"][-1])
        assert measured <= 1e-3
        return
    assert_stored_matched(expected, got, bound, exact=SCHEDULED)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Same-basis relaunch: a saved plane continues as a new run (SEP 31, phase 2).

The continued run must reproduce the uninterrupted one after the frame conversion.
Both are adaptive, so the comparison is an accuracy envelope against a tighter
reference run, not a rounding budget: the relaunch restarts the step controller and
the Adams history, so it takes other steps (``docs/dev/testing.md`` section 3).
"""

import itertools
from typing import Any, cast

import numpy as np
import pytest

from solprop.backends.registry import available
from solprop.coupling import CouplingModel, couple
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import CouplingError, DriverError
from solprop.grid import FieldGrid, RadialGrid
from solprop.processing import field_plane
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController

ENVELOPE = 1e-9
"""Accuracy envelope of a run at rtol 1e-8 against one at 1e-11, of the peak sample.

Measured on these 2 mm argon capillary cases: the uninterrupted run is 1.3e-10 from
the reference and the relaunched one 1.4e-10, against 1.3e-8 and 1.5e-8 at rtol
1e-6, so both follow the tolerance. The envelope leaves a factor of about seven.
"""


def same_basis(plane, port):
    """Return the identity model's launch of a plane into a port."""
    return couple(plane, port, model=CouplingModel("same_basis")).launch


def capillary(
    length, n_save, *, rtol=1e-8, backend="numpy_ref", radius=100e-6, **changes
):
    arguments = {
        "lambda0": 800e-9,
        "wavelength_limits": (400e-9, 1.6e-6),
        "trange": 200e-15,
        "fwhm": 20e-15,
        "energy": 20e-6,
        "modes": 2,
        "n_save": n_save,
        "plasma": False,
        "noise": None,
        "backend": backend,
        "rtol": rtol,
        **changes,
    }
    return Simulation.from_capillary(radius, length, "Ar", 1.0, **arguments)


@pytest.fixture(scope="module")
def reference():
    return field_plane(capillary(2e-3, 3, rtol=1e-11).run(progress="none"))


def relaunched(source, receiver):
    """Run 2 mm in one go and as 1 mm + 1 mm, and return both final planes."""
    whole = capillary(2e-3, 3, backend=source).run(progress="none")
    first = capillary(1e-3, 2, backend=source).run(progress="none")
    second = capillary(1e-3, 2, backend=receiver)
    launch = same_basis(field_plane(first), second.input_port())
    rest = second.with_input(launch).run(progress="none")
    return field_plane(whole), field_plane(rest)


def deviation(plane, reference):
    scale = float(np.max(np.abs(reference.samples)))
    return float(np.max(np.abs(plane.samples - reference.samples))) / scale


def test_a_relaunched_run_continues_the_uninterrupted_one(reference):
    whole, rest = relaunched("numpy_ref", "numpy_ref")
    assert deviation(whole, reference) < ENVELOPE
    assert deviation(rest, reference) < ENVELOPE
    # The inherited origin plus the second run's own frame is the whole path's: two
    # Chebyshev antiderivatives of the same constant frame, to rounding.
    assert rest.reference.delay == pytest.approx(whole.reference.delay, rel=1e-12)
    assert rest.reference.phase == pytest.approx(whole.reference.phase, abs=1e-9)
    assert rest.provenance.lineage == (
        field_plane(capillary(1e-3, 2).run(progress="none")).provenance.source,
    )


@pytest.mark.medium
@pytest.mark.parametrize(
    ("source", "receiver"),
    [
        pair
        for pair in itertools.product(available(), repeat=2)
        if pair != ("numpy_ref", "numpy_ref")
    ],
)
def test_every_cpu_source_and_receiver_pair_relaunches(source, receiver, reference):
    _, rest = relaunched(source, receiver)
    assert deviation(rest, reference) < ENVELOPE


def test_an_interior_relaunch_continues_its_own_run(reference):
    simulation = capillary(2e-3, 3)
    output = simulation.run(progress="none")
    port = simulation.input_port(1e-3)
    assert port.geometry.z == 1e-3
    assert port.frame.delay == pytest.approx(field_plane(output, 1).reference.delay)
    launch = same_basis(field_plane(output, z=1e-3), port)
    # The run's own frame up to 1 mm is the launch's reference: no origin is added.
    assert launch.origin.delay == pytest.approx(0.0, abs=1e-24)
    child = simulation.with_input(launch)
    assert child.initial_z == 1e-3
    rest = field_plane(child.run(progress="none"))
    assert deviation(rest, reference) < ENVELOPE
    with pytest.raises(DriverError, match="inside the run"):
        simulation.input_port(5e-3)


def radial(pressure=1.0, length=2e-3, backend="numpy_ref"):
    grid = FieldGrid.from_limits(2e-3, 800e-9, (600e-9, 1.2e-6), 400e-15)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-10, atol=1e-16),
        save_planes=np.linspace(0.0, length, 3)[1:],
        dz_max=2e-4,
        dz_init=2e-4,
        backend=backend,
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 6e-5
    )
    return Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 32),
        SpatialMedium("Ar", pressure),
        [],
        pulse,
        settings=settings,
    )


def test_a_radial_plane_relaunches_into_the_same_grid_and_medium():
    whole = field_plane(radial().run(progress="none"))
    first = field_plane(radial(length=1e-3).run(progress="none"))
    second = radial(length=1e-3)
    rest = field_plane(second.with_input(same_basis(first, second.input_port())).run())
    # A linear radial run is the exact flow of each column; the two stages and the
    # whole differ by the step sequence's rounding, a few hundred ulp of the phase.
    scale = float(np.max(np.abs(whole.samples)))
    assert float(np.max(np.abs(rest.samples - whole.samples))) < 1e-10 * scale
    assert rest.energy() == pytest.approx(whole.energy(), rel=1e-12)
    assert rest.reference.delay == pytest.approx(whole.reference.delay, rel=1e-12)
    denser = radial(pressure=2.0, length=1e-3)
    with pytest.raises(CouplingError, match="change of medium"):
        same_basis(first, denser.input_port())
    launch = same_basis(first, second.input_port())
    with pytest.raises(DriverError, match="metric"):
        denser.with_input(launch)


def test_a_launch_is_refused_by_any_other_simulation():
    first = field_plane(capillary(1e-3, 2).run(progress="none"))
    second = capillary(1e-3, 2)
    launch = same_basis(first, second.input_port())
    wider = capillary(1e-3, 2, radius=110e-6)
    with pytest.raises(DriverError, match="basis"):
        wider.with_input(launch)
    with pytest.raises(CouplingError, match="not a relaunch"):
        same_basis(first, wider.input_port())
    longer = capillary(1e-3, 2, trange=400e-15)
    with pytest.raises(DriverError, match="lattice"):
        longer.with_input(launch)
    with pytest.raises(DriverError, match="initial_z"):
        second.with_input(launch, initial_z=0.0)
    with pytest.raises(DriverError, match="recorded by the run itself"):
        second.with_input(launch, configuration={"coupling": "mine"})


def test_a_launch_records_its_provenance_and_draws_no_noise():
    first = field_plane(capillary(1e-3, 2).run(progress="none"))
    noisy = capillary(1e-3, 2, noise="input", seed=5)
    assert noisy.noise is not None
    launch = same_basis(first, noisy.input_port())
    child = noisy.with_input(launch)
    assert child.noise is None
    record = cast("dict[str, Any]", child.metadata().configuration["coupling"])
    assert record["model"] == "same_basis"
    assert record["port"] == launch.port.identity
    assert record["source"]["plane"]["source"] == first.provenance.source
    reference = child.metadata().reference
    assert reference is not None
    assert reference["origin"] == first.reference.minus(launch.port.frame).record()
    assert reference["parent"] == record["source"]
    kept = noisy.with_input(launch, noise=noisy.noise)
    assert kept.noise is noisy.noise
    # A plain array input into a coupled run is not that coupling any more.
    plain = child.with_input(launch.internal())
    assert "coupling" not in plain.configuration
    grandchild_port = plain.input_port()
    assert grandchild_port.identity == launch.port.identity

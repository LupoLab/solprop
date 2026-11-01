# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The problems that take the z table run on the native CPU build (SEP 29 section 6.6).

Each way a run reaches the general z table, through the public interface, against the
reference: a multimode capillary taper on the adiabatic opt-out
(``taper_coupling=False``), whose expansion does not resolve the kinked taper; a
radial run through three species on independent density paths, more than the
two-parameter expansion takes; optmat's dry air falling to vacuum, three populations on
a modal guide; and a ``Batch`` of moving-dispersion fibres. The moving dispersion and
the temperature-path fill are in ``tests/driver/test_specified_paths.py`` and
``tests/driver/test_temperature_paths.py``, a forced table and a table prefactor in
``tests/backends/test_native_z_table.py``. Ten fixed steps of each march, the native
run matched to the reference's within the march's budget (``tests/march_bounds.py``):
the same method, schedule and history, so they differ by rounding alone. Dense planes
and the modified noise read the table's phase too.
"""

import copy
import dataclasses
import warnings
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any, cast

import numpy as np
import pytest

from driver.test_specified_paths import GRID, PULSE, linear_walkoff
from driver.test_specified_paths import LENGTH as FIBRE_LENGTH
from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import (
    FIXED_ORDER,
    TIGHTER,
    assert_in_envelope,
    assert_matched,
    simulation_bound,
)
from optmat_parity_cases import variant
from solprop.backends import Backend, select, select_batch
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import Mixture, gamma3
from solprop.profiles import Linear, PiecewiseLinear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import (
    FixedSchedule,
    LowStorageRungeKutta,
    Multistep,
    PIController,
    RungeKutta,
)

pytestmark = pytest.mark.medium
pytest.importorskip("solprop_native")

STEPS = 10

MARCHES = pytest.mark.parametrize(
    "integrator",
    [FIXED_ORDER, RungeKutta(), LowStorageRungeKutta()],
    ids=["adams", "runge_kutta", "low_storage"],
)
"""Each march on the prescribed schedule; the exponential Adams method carries a
varying operator's compensation, whose rate the native derivative table supplies."""


@pytest.fixture(autouse=True)
def _quiet():
    """Silence the gas indices' extrapolation notice and the flow's step advice."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        warnings.filterwarnings("ignore", message=".*explicit residual is the limit")
        yield


def on(simulation, backend):
    """Return a shallow copy of `simulation` that runs on `backend`.

    The translation is shared, so both runs step the same problem.
    """
    other = copy.copy(simulation)
    other.settings = dataclasses.replace(simulation.settings, backend=backend)
    return other


def assert_native_matches(simulation):
    """Run `simulation` on the native backend and the reference; assert the match."""
    problem = simulation.translate()
    assert not problem.device_ready
    native = on(simulation, "native").run(progress="none").fields("fft")
    reference = on(simulation, "numpy_ref").run(progress="none").fields("fft")
    bound = simulation_bound(simulation, backend="native")
    assert_matched(native, reference, bound, what="the z table")


@MARCHES
def test_a_multimode_taper_without_its_coupling(integrator):
    # Four channels of a metre's taper, ten 0.1 m steps of a nanojoule: the adiabatic
    # route's operator kinks with the taper and takes the z table. The collar is
    # integrated exactly, so that the explicit marches are stable at these steps.
    taper = PiecewiseLinear(
        np.array([0.0, 0.5, 0.7, 1.0]), np.array([100e-6, 100e-6, 60e-6, 60e-6])
    )
    simulation = Simulation.from_capillary(
        taper,
        1.0,
        "He",
        1.5,
        taper_coupling=False,
        wavelength=800e-9,
        fwhm=10e-15,
        energy=1e-9,
        # The former grid, on which the operator's expansion does not resolve.
        grid=FieldGrid.from_limits(1.0, 800e-9, (90e-9, 3e-6), 250e-15, **LEGACY_GRID),
        modes=4,
        n_save=2,
        schedule=FixedSchedule(np.full(STEPS, 1.0 / STEPS)),
        integrator=integrator,
        absorber_scheme="split_step",
        backend="numpy_ref",
        noise=None,
    )
    assert simulation.translate().linear.phi_representation == "table"
    assert_native_matches(simulation)


@MARCHES
def test_a_radial_run_through_three_independent_species(integrator):
    # Three density paths exceed the two-parameter expansion on the radial route, so
    # both the operator and the prefactor take the z table.
    length = 1e-2
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (400e-9, 2e-6), 200e-15, **LEGACY_GRID
    )
    fill = Mixture.fill({"He": 0.4, "Ar": 0.3, "Ne": 0.3}, 1.0)
    fill = fill.with_densities(
        {
            label: Linear((index + 1.0) * 1e24, (index + 1.5) * 1e24, length)
            for index, label in enumerate(fill.labels)
        }
    )
    simulation = Simulation.from_radial(
        grid,
        RadialGrid(1e-3, 16, collar_fraction=0.0),
        SpatialMedium(fill),
        [
            KerrResponse(gamma3(label, wavelength=800e-9), thg=False, species=label)
            for label in fill.labels
        ],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 1e-4),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[length],
            dz_max=length / STEPS * 1.0000001,
            dz_init=length / STEPS,
            schedule=FixedSchedule(np.full(STEPS, length / STEPS)),
            integrator=integrator,
            backend="numpy_ref",
            # No collar: ten fixed steps are refused beside the default one.
            absorber=bare_absorber(grid),
        ),
        noise=None,
    )
    problem = simulation.translate()
    assert problem.linear.phi_representation == "table"
    assert problem.responses.prefactor_representation == "table"
    assert_native_matches(simulation)


@pytest.mark.parametrize(
    "integrator", [RungeKutta(), LowStorageRungeKutta()], ids=["runge_kutta", "low"]
)
def test_dry_air_falling_to_vacuum(integrator):
    # optmat's dry air on three independent density paths; its Adams run is the
    # parity ladder of tests/backends/test_optmat_raman_parity.py.
    case = variant("air_vacuum", integrator=integrator)
    assert case.simulation is not None
    assert case.problem.linear.phi_representation == "table"
    assert_native_matches(case.simulation)


def fibre(index, backend):
    """Return member `index` of a moving-dispersion batch, its own area and power."""
    grid = EnvelopeGrid.from_limits(FIBRE_LENGTH, 835e-9, (600e-9, 1300e-9), 2e-12)
    schedule = FixedSchedule.uniform(
        0.0, FIBRE_LENGTH, FIBRE_LENGTH / 4, save_planes=[FIBRE_LENGTH]
    )
    return Simulation.from_gnlse(
        0.1 * (index + 1),
        FIBRE_LENGTH,
        fibre=linear_walkoff(grid, areas=(1e-10, (1.2 + 0.3 * index) * 1e-10)),
        shock=False,
        save_planes=[FIBRE_LENGTH],
        noise=None,
        schedule=schedule,
        integrator=RungeKutta(),
        backend=backend,
        # No collar: four fixed steps are refused beside the default one.
        absorber=False,
        **GRID,
        **PULSE,
    )


def loaded(simulation, cleanup):
    """Return the simulation's backend, loaded at the entrance and closed at exit."""
    problem = simulation.translate()
    owner = select(
        simulation.settings.backend,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    cleanup.callback(owner.close)
    owner.set_initial(problem.initial.field0, simulation.noise)
    return owner


def test_a_batch_of_z_table_members():
    # The native Batch prepares each member's trial on its own tables and fuses the
    # accepted update, as it does for parameter expansions.
    with ExitStack() as cleanup:
        members = [fibre(index, "native") for index in range(3)]
        assert not members[0].translate().device_ready
        batch = select_batch(
            cast(Sequence[Backend], [loaded(one, cleanup) for one in members])
        )
        cleanup.callback(batch.close)
        assert type(batch).__name__ == "NativeBatchBackend"
        batch.propagate_to(FIBRE_LENGTH)
        observed = batch.observe_members()
        for index, (member, ours) in enumerate(zip(members, observed, strict=True)):
            reference: Any = loaded(on(member, "numpy_ref"), cleanup)
            reference.propagate_to(FIBRE_LENGTH)
            bound = simulation_bound(member, backend="native")
            assert_matched(
                ours.require_field(),
                reference.observe().require_field(),
                bound,
                what=f"member {index}",
            )


DENSE = pytest.mark.parametrize(
    "integrator",
    [Multistep(), Multistep(formulation="interaction"), RungeKutta()],
    ids=["adams", "interaction", "runge_kutta"],
)


def walk_off(backend, integrator, rtol=1e-8, **options):
    """Return the moving-dispersion fibre of a nanojoule, adaptive unless scheduled."""
    grid = EnvelopeGrid.from_limits(FIBRE_LENGTH, 835e-9, (600e-9, 1300e-9), 2e-12)
    options.setdefault("noise", None)
    return Simulation.from_gnlse(
        0.1,
        FIBRE_LENGTH,
        fibre=linear_walkoff(grid),
        shock=False,
        integrator=integrator,
        rtol=rtol,
        backend=backend,
        **GRID,
        **(PULSE | {"energy": 1e-9}),
        **options,
    )


@DENSE
def test_dense_planes_interpolate_through_the_table(integrator):
    # A plane inside an accepted step is interpolated through the step's own phase,
    # which the native backend reads from the table's slots. The runs are adaptive,
    # so the native run is held to the reference run's accuracy envelope against
    # the same problem at a hundredth of the tolerance (SEP 25 section 4.4).
    options = {"n_save": 7, "noise": None, "interpolate_planes": True}
    native = walk_off("native", integrator, **options).run(progress="none")
    reference = walk_off("numpy_ref", integrator, **options).run(progress="none")
    tight = walk_off("numpy_ref", integrator, rtol=1e-8 / TIGHTER, **options).run(
        progress="none"
    )
    assert_in_envelope(
        [native.fields("fft")],
        tight.fields("fft"),
        reference.fields("fft"),
        what="dense planes on the z table",
    )


@pytest.mark.parametrize(
    "integrator", [FIXED_ORDER, RungeKutta()], ids=["adams", "runge_kutta"]
)
def test_the_noise_is_advanced_by_the_tables_phase(integrator):
    # The modified noise is advanced by Im Phi at each stage and plane, read from
    # the table's slots: ten fixed steps, matched to the reference.
    planes = [FIBRE_LENGTH * index / 5 for index in range(1, 6)]
    schedule = FixedSchedule.uniform(
        0.0, FIBRE_LENGTH, FIBRE_LENGTH / STEPS, save_planes=planes
    )
    simulation = walk_off(
        "numpy_ref",
        integrator,
        save_planes=planes,
        schedule=schedule,
        noise="modified",
        seed=3,
        # No collar: ten fixed steps are refused beside the default one.
        absorber=False,
    )
    assert_native_matches(simulation)

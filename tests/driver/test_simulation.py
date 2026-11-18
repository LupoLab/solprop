# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The simulation object: translating once, recording what it is, feeding a backend."""

import contextlib
import dataclasses
import math
import pickle
import re
import subprocess
import sys
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from builders import grid_at
from solprop._identity import canonical_json
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.driver import settings as settings_module
from solprop.driver import simulation as simulation_module
from solprop.exceptions import DriverError, MaterialRangeWarning
from solprop.nonlinear.gnlse import problem_from
from solprop.output.serialisation import serialise
from solprop.profiles import Constant, PiecewiseLinear, PressureGradient
from solprop.pulses import SechPulse
from solprop.stats import StatsSpec
from solprop.stepper import PIController

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

# A fibre with third-order dispersion, so beta2 crosses zero inside the band: with
# beta2(omega) = beta_2 + beta_3 (omega - omega_ref), the zero sits at
# omega_ref - beta_2 / beta_3, which is put at 1000 nm below.
ZERO_DISPERSION_WAVELENGTH = 1000e-9
BETA_2 = 2.0e-26  # s^2/m, normal at the reference


def dispersive_betas(wavelength):
    """Return Taylor coefficients whose beta2 vanishes at 1000 nm."""
    omega_ref = 2.0 * math.pi * C / wavelength
    detuning = 2.0 * math.pi * C / ZERO_DISPERSION_WAVELENGTH - omega_ref
    return [0.0, 0.0, BETA_2, -BETA_2 / detuning]


def test_the_problem_is_translated_once_for_two_runs(build_simulation, monkeypatch):
    calls = []
    original = simulation_module.assemble

    def counted(*arguments, **keywords):
        calls.append(1)
        return original(*arguments, **keywords)

    monkeypatch.setattr(simulation_module, "assemble", counted)
    simulation = build_simulation(n_save=2)
    simulation.run()
    simulation.run()
    assert len(calls) == 1


@pytest.mark.parametrize("first", ["translate", "run_request"])
def test_the_run_request_is_the_one_translated_beside_the_problem(
    build_simulation, monkeypatch, first
):
    """Either call translates, once; the other returns what that translation built."""
    calls = []
    original = simulation_module.assemble

    def counted(*arguments, **keywords):
        calls.append(1)
        return original(*arguments, **keywords)

    monkeypatch.setattr(simulation_module, "assemble", counted)
    simulation = build_simulation(n_save=2)
    getattr(simulation, first)()
    request = simulation.run_request()
    problem = simulation.translate()
    assert len(calls) == 1
    assert simulation.run_request() is request
    assert request.observation.start_plane == 0
    np.testing.assert_array_equal(
        request.observation.save_planes, simulation.settings.save_planes
    )
    # A run adapts a copy; the cached request stays the fresh run's.
    assert request.for_run(start_plane=1) is not request
    assert simulation.run_request().observation.start_plane == 0
    assert simulation.translate() is problem


def test_a_derived_simulation_has_its_own_run_request(build_simulation):
    """An interior launch's request holds the planes beyond it, its parent's all."""
    simulation = build_simulation(n_save=3)
    planes = simulation.run_request().observation.save_planes
    launch_z = 0.5 * float(planes[0] + planes[1])
    derived = simulation.with_input(simulation._field0, initial_z=launch_z)
    request = derived.run_request()
    assert request is derived.run_request()
    assert request.observation.start_plane == 0
    np.testing.assert_array_equal(request.observation.save_planes, planes[1:])
    assert simulation.run_request().observation.save_planes is planes


def test_settings_assigned_before_the_first_translation_take_effect(build_simulation):
    """A replacement before :meth:`translate` reaches the problem."""
    simulation = build_simulation(n_save=2)
    simulation.settings = dataclasses.replace(simulation.settings, dz_max=1e-4)
    assert simulation.settings.dz_max == 1e-4
    assert simulation.run_request().stepper.dz_max == 1e-4


@pytest.mark.parametrize("first", ["translate", "run"])
def test_settings_assigned_after_the_first_translation_are_refused(
    build_simulation, first
):
    """The cached problem would ignore new settings, so assigning raises."""
    simulation = build_simulation(n_save=2)
    getattr(simulation, first)()
    settings = simulation.settings
    with pytest.raises(DriverError, match=r"already translated.*new Simulation"):
        simulation.settings = dataclasses.replace(settings, shock=not settings.shock)
    with pytest.raises(DriverError, match=r"a new save_planes, dz_max would be"):
        simulation.settings = dataclasses.replace(
            settings, dz_max=1e-4, save_planes=[1.234e-4], backend="native"
        )
    assert simulation.settings is settings


def test_the_execution_settings_change_after_translation_and_the_refusal_names_them(
    build_simulation,
):
    simulation = build_simulation(n_save=2)
    simulation.translate()
    simulation.settings = dataclasses.replace(simulation.settings, node_budget=2**20)
    assert simulation.settings.node_budget == 2**20
    allowed = (
        "only backend, node_budget, resume_history, tdse_execution, fields, "
        "field_planes and boundary_check may change after translation"
    )
    with pytest.raises(DriverError, match=re.escape(allowed)):
        simulation.settings = dataclasses.replace(simulation.settings, dz_max=1e-4)


@pytest.mark.parametrize("name", sorted(simulation_module._FIXED_INPUTS))
def test_assigning_what_a_simulation_is_built_from_is_refused(build_simulation, name):
    """The input and the record are built from these at construction, so a new
    value would run the old input and record the new one; the refusal names the
    methods that derive a simulation instead."""
    simulation = build_simulation(n_save=2)
    before = getattr(simulation, name)
    with pytest.raises(AttributeError, match=r"with_input\(\.\.\.\).*new Simulation"):
        setattr(simulation, name, before)
    assert getattr(simulation, name) is before or name in ("profiles", "configuration")


def test_the_recorded_mappings_are_copies(build_simulation):
    """Changing the configuration or profiles in place would change the record alone,
    so each read is a plain copy: picklable, and changing it changes nothing."""
    simulation = build_simulation(n_save=2)
    configuration = simulation.configuration
    configuration["label"] = "changed"
    profiles = simulation.profiles
    profiles["density"] = Constant(1.0)
    assert "label" not in simulation.configuration
    assert "density" not in simulation.profiles
    assert "label" not in simulation.metadata().configuration
    pickle.dumps((simulation.configuration, simulation.profiles))


def test_with_input_derives_the_run_a_new_pulse_makes(build_simulation):
    """The way to a new input: the derived run runs and records it, the original
    keeps its own."""
    simulation = build_simulation(n_save=2)
    stronger = dataclasses.replace(
        simulation.initial,
        peak_power=2 * simulation.initial.peak_power,  # pyright: ignore[reportAttributeAccessIssue]
    )
    derived = simulation.with_input(stronger)
    assert derived.initial is stronger
    assert derived.metadata().configuration_hash() != (
        simulation.metadata().configuration_hash()
    )
    # Twice the peak power is twice the energy at the entrance; exact up to rounding.
    np.testing.assert_allclose(
        np.sum(np.abs(derived._field0) ** 2),
        2 * np.sum(np.abs(simulation._field0) ** 2),
        rtol=1e-12,
    )


def test_the_backend_alone_may_change_after_translation(build_simulation):
    """The backend is read when the run executes, so changing it is not ignored."""
    simulation = build_simulation(n_save=2)
    problem = simulation.translate()
    # Equal values in new objects (save_planes is rebuilt by replace) are the same.
    simulation.settings = dataclasses.replace(
        simulation.settings,
        backend="numpy_ref",
        save_planes=list(simulation.settings.save_planes),
    )
    assert simulation.settings.backend == "numpy_ref"
    handler = simulation.run(progress="none")
    assert simulation.translate() is problem
    assert handler.metadata is not None and handler.metadata.backend is not None
    assert handler.metadata.backend.name == "numpy_ref"


def test_a_setting_that_cannot_be_compared_is_not_the_same():
    """A container of arrays has no truth value; it is never taken as unchanged."""
    same = settings_module._same_setting
    assert same((np.ones(2),), (np.ones(2),)) is False
    assert same(np.ones(2), [1.0, 1.0]) and not same(np.ones(2), np.ones(3))
    assert same(1e-3, 1e-3) and not same(1e-3, 2e-3)


def test_the_configuration_hash_is_stable_and_changes_with_gamma(build_simulation):
    """Two simulations built the same way are the same run; a stronger one is not.

    Stability across objects is what a resume from another process rests on: the digest
    covers the serialised mode, and a mode built from closures would carry their memory
    addresses instead of their coefficients.
    """
    one = build_simulation(n_save=2)
    again = build_simulation(n_save=2)
    assert one.metadata().configuration_hash() == again.metadata().configuration_hash()
    stronger = build_simulation(n_save=2, gamma=0.22)
    assert one.metadata().configuration_hash() != (
        stronger.metadata().configuration_hash()
    )
    changed = one.metadata().differences(stronger.metadata())
    assert any("gnlse.gamma" in path for path in changed), changed


def test_the_metadata_records_the_planes_the_statistics_and_the_stepper(
    build_simulation,
):
    metadata = build_simulation(n_save=3).metadata()
    assert metadata.configuration["save_planes"].size == 2
    assert metadata.configuration["statistics"].names[0] == "omega0"
    assert metadata.stepper.tableau == "DOPRI5"
    assert metadata.stepper.limits.dz_max > 0.0
    assert metadata.noise is None


def test_the_modes_and_the_profiles_reach_the_backend(narrow_grid, soliton):
    """A zdw needs the modes and a density needs a profile; both come from here.

    The kernel contract carries neither, so the driver passes them to
    ``registry.select`` and the backend hands them to ``stats.collect``. Both statistics
    are checked against their own closed form.
    """
    wavelength = soliton.wavelength
    fibre = problem_from(
        dispersive_betas(wavelength),
        omega_ref=2.0 * math.pi * C / wavelength,
        gamma=soliton.gamma,
        shock=False,
    )
    pulse = SechPulse(
        wavelength=wavelength, width=soliton.tau0, peak_power=soliton.peak_power(1)
    )
    simulation = Simulation(
        grid_at(narrow_grid, 0.005),
        fibre.mode,
        [fibre.kerr],
        pulse,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-18),
            save_planes=[0.005],
            dz_max=0.005,
            dz_init=1e-4,
            density=1.0,
            shock=False,
            prefactor_index=1.0,
            stats=StatsSpec(names=("zdw", "density")),
        ),
        profiles={"density": Constant(2.5e25)},
    )
    handler = simulation.run()
    # brentq's own tolerance on the root is 2e-12 relative in omega, and beta2 here is
    # exactly linear in omega, so the only error is the root finder's: a picometre is
    # four orders above what it achieves and still far below any physical scale.
    assert handler["stats/zdw"][-1] == pytest.approx(
        ZERO_DISPERSION_WAVELENGTH, abs=1e-12
    )
    assert handler["stats/density"][-1] == 2.5e25


def test_a_built_spectrum_is_accepted_as_the_input(narrow_grid, soliton):
    fibre = problem_from(
        [0.0, 0.0, soliton.beta2],
        omega_ref=narrow_grid.omega0,
        gamma=soliton.gamma,
        shock=False,
    )
    spectrum = SechPulse(
        wavelength=soliton.wavelength,
        width=soliton.tau0,
        peak_power=soliton.peak_power(1),
    ).build(narrow_grid)
    simulation = Simulation(
        grid_at(narrow_grid, 0.005),
        fibre.mode,
        [fibre.kerr],
        spectrum,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-18),
            save_planes=[0.005],
            dz_max=0.005,
            dz_init=1e-4,
            density=1.0,
            shock=False,
            prefactor_index=1.0,
        ),
    )
    handler = simulation.run()
    assert handler.n_saved == 2
    assert np.array_equal(simulation.translate().initial.field0, spectrum)


def test_an_input_that_is_neither_a_pulse_nor_a_spectrum_is_refused(
    narrow_grid, soliton
):
    fibre = problem_from(
        [0.0, 0.0, soliton.beta2], omega_ref=narrow_grid.omega0, gamma=soliton.gamma
    )
    # A runtime-checkable protocol is satisfied by the presence of `build`, so this
    # object passes the annotation at run time and reaches the guard that checks it is
    # callable; it is typed loosely so that the static checker lets it through too.
    not_a_pulse: Any = SimpleNamespace(build="a pulse, honest")
    with pytest.raises(DriverError, match="build\\(grid\\) method"):
        Simulation(
            grid_at(narrow_grid, 0.005),
            fibre.mode,
            [fibre.kerr],
            not_a_pulse,
            settings=SimulationSettings(
                controller=PIController(rtol=1e-8, atol=1e-18),
                save_planes=[0.005],
                dz_max=0.005,
                dz_init=1e-4,
            ),
        )


def test_from_capillary_needs_a_grid_or_the_keywords_to_build_one():
    with pytest.raises(DriverError, match="wavelength_limits and trange"):
        Simulation.from_capillary(125e-6, 0.3, "He", 1.0, wavelength=800e-9, noise=None)


# ----------------------------------------------------- the capillary constructor


CAPILLARY = {
    "wavelength": 800e-9,
    "wavelength_limits": (200e-9, 2e-6),
    "trange": 100e-15,
    "fwhm": 10e-15,
    "energy": 20e-6,
    "n_r": 8,
    "n_save": 2,
}


@contextlib.contextmanager
def quiet():
    """Silence the gas index's range notice, which every capillary band triggers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        yield


def capillary(pressure=1.0, **overrides):
    """Build a small capillary simulation."""
    with quiet():
        return Simulation.from_capillary(
            125e-6, 1e-3, "He", pressure, **({"noise": None, **CAPILLARY | overrides})
        )


def digest(simulation):
    """Return a simulation's configuration hash, which translating it produces."""
    with quiet():
        return simulation.metadata().configuration_hash()


def test_a_capillary_configuration_hashes_the_same_way_twice():
    assert digest(capillary()) == digest(capillary())


@pytest.mark.child_runs_auto
def test_a_capillary_configuration_hashes_the_same_way_in_a_new_process():
    """What a resume from another process rests on: no address ends up in the digest.

    Every object the configuration carries -- the modes, the pressure and density
    paths, the responses, the rate object and the pulse -- has to serialise to its own
    data rather than to a ``repr`` holding a memory address
    (:func:`solprop.output.serialisation.serialise`).
    """
    program = (
        "import warnings, sys\n"
        "from solprop.exceptions import MaterialRangeWarning\n"
        "warnings.simplefilter('ignore', MaterialRangeWarning)\n"
        "from solprop.driver import Simulation\n"
        "simulation = Simulation.from_capillary(\n"
        f"    125e-6, 1e-3, 'He', 1.0, noise=None, **{CAPILLARY!r}\n"
        ")\n"
        "sys.stdout.write(simulation.metadata().configuration_hash())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == digest(capillary())


def test_a_capillary_configuration_changes_with_what_it_describes():
    """A different fill is a different run, and the digest says which field moved."""
    one, denser = capillary(), capillary(2.0)
    assert digest(one) != digest(denser)
    with quiet():
        changed = one.metadata().differences(denser.metadata())
    assert changed


def test_the_modes_the_profiles_and_the_grid_reach_the_backend():
    """The backend is handed what the kernel contract cannot carry (ADR-012).

    Read through the statistics rather than through the call: ``peak_intensity`` needs
    the mode set to synthesise the field on the axis and ``density`` needs the
    z-profile, and neither is in the contract, so a run that reports both got both.
    """
    simulation = capillary(modes=2)
    with quiet():
        problem = simulation.translate()
    assert problem.shape[0] == 2
    assert problem.transform.synthesis.shape[1] == 2
    assert set(simulation.profiles) == {"density", "pressure", "core_radius"}
    with quiet():
        output = simulation.run()
    assert float(output["stats/peak_intensity"][0]) > 0.0
    assert float(output["stats/density"][0]) > 0.0
    assert output["field"].shape[1] == 2


# ------------------------------------------ what a Simulation keeps of its caller's


def _profile_tables():
    """Return the arrays of a density table and a pressure table, fresh each call."""
    return {
        "density_positions": np.array([0.0, 0.0025, 0.005]),
        "density_values": np.array([2.5e25, 2.0e25, 1.5e25]),
        "pressure_positions": np.array([0.0, 0.005]),
        "pressure_values": np.array([1.0, 0.5]),
    }


def _configuration():
    """Return a caller's provenance with a list, a nested dict and an array in it."""
    return {
        "coefficients": np.array([1.0, 2.0, 3.0]),
        "labels": ["scan", "point"],
        "scan": {"delays": [1e-15, 2e-15], "pair": (0.5, np.array([4.0]))},
        "tags": {"a", "b"},
    }


def _caller_simulation(narrow_grid, soliton, tables, configuration):
    """Build a direct Simulation whose density and pressure statistics read profiles."""
    fibre = problem_from(
        [0.0, 0.0, soliton.beta2],
        omega_ref=narrow_grid.omega0,
        gamma=soliton.gamma,
        shock=False,
    )
    profiles = {
        "density": PiecewiseLinear(
            tables["density_positions"], tables["density_values"]
        ),
        "pressure": PressureGradient(
            positions=tables["pressure_positions"],
            pressures=tables["pressure_values"],
            floor=0.0,
        ),
    }
    return Simulation(
        narrow_grid,
        fibre.mode,
        [fibre.kerr],
        SechPulse(
            wavelength=soliton.wavelength,
            width=soliton.tau0,
            peak_power=soliton.peak_power(1),
        ),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-8, atol=1e-18),
            save_planes=[0.0025, 0.005],
            dz_max=0.005,
            dz_init=1e-4,
            density=1.0,
            shock=False,
            prefactor_index=1.0,
            stats=StatsSpec(names=("density", "pressure")),
        ),
        profiles=profiles,
        configuration=configuration,
    )


def _change_everything(tables, configuration):
    """Change, in place, every array, list, dict and set the caller passed."""
    for values in tables.values():
        np.multiply(values, 2.0, out=values)
    configuration["coefficients"][0] = -1.0
    configuration["labels"].append("changed")
    configuration["scan"]["delays"][0] = 9e-15
    configuration["scan"]["pair"][1][0] = -4.0
    configuration["scan"]["added"] = True
    configuration["tags"].add("c")
    configuration["late"] = "key"


@pytest.mark.parametrize("translated_first", [False, True], ids=["lazy", "translated"])
def test_changing_what_the_caller_passed_changes_neither_the_run_nor_its_record(
    narrow_grid, soliton, translated_first
):
    """Profiles and configuration are read when the simulation is built (issue #471).

    The run's density and pressure statistics are computed from the profiles, and the
    record holds both them and the configuration, so a simulation that shared the
    caller's tables, lists or arrays would follow a change made after construction.
    The reference is the same simulation built from untouched copies.
    """
    tables, configuration = _profile_tables(), _configuration()
    simulation = _caller_simulation(narrow_grid, soliton, tables, configuration)
    if translated_first:
        simulation.translate()
    _change_everything(tables, configuration)
    reference = _caller_simulation(
        narrow_grid, soliton, _profile_tables(), _configuration()
    )
    changed = simulation.metadata().differences(reference.metadata())
    assert not changed, changed
    assert (
        simulation.metadata().configuration_hash()
        == reference.metadata().configuration_hash()
    )
    assert simulation.configuration["labels"] == ["scan", "point"]
    kept = simulation.configuration["coefficients"]
    assert isinstance(kept, np.ndarray)
    assert not kept.flags.writeable
    output, expected = simulation.run(), reference.run()
    # Exact: both runs evaluate the same profile tables at the same planes.
    for name in ("stats/density", "stats/pressure"):
        np.testing.assert_array_equal(output[name], expected[name])
    # The statistic is the backend's Chebyshev interpolant of the table, exact to a
    # few roundings; the doubled table would read 5e25.
    assert float(output["stats/density"][0]) == pytest.approx(2.5e25, rel=1e-12)


def test_a_derived_runs_configuration_is_copied_from_the_caller(narrow_grid, soliton):
    """`with_input` copies its caller's keys as the constructor does (issue #471)."""
    simulation = _caller_simulation(
        narrow_grid, soliton, _profile_tables(), _configuration()
    )
    pulse = SechPulse(
        wavelength=soliton.wavelength,
        width=soliton.tau0,
        peak_power=soliton.peak_power(1),
    )
    keys = {"delay": np.array([1e-15]), "label": ["first"]}
    derived = simulation.with_input(pulse, configuration=keys)
    keys["delay"][0] = 5e-15
    keys["label"].append("second")
    reference = simulation.with_input(
        pulse, configuration={"delay": np.array([1e-15]), "label": ["first"]}
    )
    assert not derived.metadata().differences(reference.metadata())
    assert (
        derived.metadata().configuration_hash()
        == reference.metadata().configuration_hash()
    )


def test_copying_the_callers_inputs_leaves_the_record_and_its_hash_unchanged(
    narrow_grid, soliton
):
    """The copies serialise as the caller's objects did, so no digest moves.

    Before issue #471 a simulation kept the caller's mapping of profiles and the
    configuration's values themselves; a run recorded that way must have the record
    and the configuration hash a run recorded from the copies has, or a resume of a
    file written before the copy would be refused.
    """
    tables, configuration = _profile_tables(), _configuration()
    copied = _caller_simulation(narrow_grid, soliton, tables, configuration)
    shared = _caller_simulation(narrow_grid, soliton, tables, configuration)
    # How a simulation kept them before the copy.
    shared._profiles = dict(shared._profiles)
    shared._configuration = dict(configuration)
    assert shared.configuration["coefficients"] is configuration["coefficients"]
    assert copied.configuration["coefficients"] is not configuration["coefficients"]
    record, before = copied.metadata(), shared.metadata()
    assert record.configuration_hash() == before.configuration_hash()
    assert not record.differences(before)
    assert canonical_json(record.hashed()) == canonical_json(before.hashed())
    assert serialise(record.configuration) == serialise(before.configuration)


def _with_components(simulation, components):
    """Give a simulation's untranslated settings these transverse components."""
    simulation.settings = dataclasses.replace(
        simulation.settings, components=components
    )
    return simulation


def test_the_settings_keep_a_copy_of_the_callers_components():
    """Changing the caller's list after building the settings changes nothing (#471).

    The simulation translates its settings lazily, so settings that shared the list
    would let a change made between construction and translation reach the run.
    """
    given = ["x"]
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-18),
        save_planes=[0.005],
        dz_max=0.005,
        dz_init=1e-4,
        components=given,
    )
    given.append("y")
    assert settings.components == ("x",)
    caller = ["x"]
    simulation = _with_components(capillary(modes=2), caller)
    caller.append("y")
    reference = _with_components(capillary(modes=2), ("x",))
    with quiet():
        assert simulation.translate().shape == reference.translate().shape
        assert not simulation.metadata().differences(reference.metadata())
    assert digest(simulation) == digest(reference)


def test_components_given_as_a_list_record_as_a_tuple_does():
    """The copy is a tuple; a run given a list records what one given a tuple does.

    Translation resolves the components as a tuple whatever it is given, so the
    record and its configuration hash are those of the components themselves.
    """
    listed = _with_components(capillary(modes=2), ["x"])
    paired = _with_components(capillary(modes=2), ("x",))
    with quiet():
        record, expected = listed.metadata(), paired.metadata()
    assert record.configuration_hash() == expected.configuration_hash()
    assert canonical_json(record.hashed()) == canonical_json(expected.hashed())

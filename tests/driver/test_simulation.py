# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The simulation object: translating once, recording what it is, feeding a backend."""

import contextlib
import dataclasses
import math
import subprocess
import sys
import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.driver import simulation as simulation_module
from solprop.exceptions import DriverError, MaterialRangeWarning
from solprop.nonlinear.gnlse import problem_from
from solprop.profiles import Constant
from solprop.pulses import SechPulse
from solprop.stats import StatsSpec
from solprop.stepper import PIController

# A fibre with third-order dispersion, so beta2 crosses zero inside the band: with
# beta2(omega) = beta_2 + beta_3 (omega - omega_ref), the zero sits at
# omega_ref - beta_2 / beta_3, which is put at 1000 nm below.
ZERO_DISPERSION_WAVELENGTH = 1000e-9
BETA_2 = 2.0e-26  # s^2/m, normal at the reference


def dispersive_betas(lambda0):
    """Return Taylor coefficients whose beta2 vanishes at 1000 nm."""
    omega_ref = 2.0 * math.pi * C / lambda0
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


def test_settings_assigned_before_the_first_translation_take_effect(build_simulation):
    """A replacement before :meth:`translate` reaches the problem."""
    simulation = build_simulation(n_save=2)
    simulation.settings = dataclasses.replace(simulation.settings, dz_max=1e-4)
    assert simulation.settings.dz_max == 1e-4
    assert simulation.translate().stepper.dz_max == 1e-4


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
            settings, dz_max=1e-4, save_planes=[1.234e-4], backend="numba_cpu"
        )
    assert simulation.settings is settings


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
    same = simulation_module._same_setting
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
    lambda0 = soliton.wavelength
    fibre = problem_from(
        dispersive_betas(lambda0),
        omega_ref=2.0 * math.pi * C / lambda0,
        gamma=soliton.gamma,
        shock=False,
    )
    pulse = SechPulse(
        wavelength=lambda0, width=soliton.tau0, peak_power=soliton.peak_power(1)
    )
    simulation = Simulation(
        narrow_grid,
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
        narrow_grid,
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
    assert np.array_equal(simulation.translate().initial.field0[0], spectrum)


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
            narrow_grid,
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
        Simulation.from_capillary(125e-6, 0.3, "He", 1.0, lambda0=800e-9, noise=None)


# ----------------------------------------------------- the capillary constructor


CAPILLARY = {
    "lambda0": 800e-9,
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


def test_a_capillary_configuration_hashes_the_same_way_in_a_new_process():
    """What a resume from another process rests on: no address ends up in the digest.

    Every object the configuration carries -- the modes, the pressure and density
    paths, the responses, the rate object and the pulse -- has to serialise to its own
    data rather than to a ``repr`` holding a memory address
    (:func:`solprop.output.schema.serialise`).
    """
    program = (
        "import warnings, sys\n"
        "from solprop.exceptions import MaterialRangeWarning\n"
        "warnings.simplefilter('ignore', MaterialRangeWarning)\n"
        "from solprop.driver import Simulation\n"
        "simulation = Simulation.from_capillary(\n"
        f"    125e-6, 1e-3, 'He', 1.0, **{CAPILLARY!r}\n"
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
    assert problem.shape[1] == 2
    assert problem.transform.synthesis.shape[1] == 2
    assert set(simulation.profiles) == {"density", "pressure", "core_radius"}
    with quiet():
        output = simulation.run()
    assert float(output["stats/peak_intensity"][0]) > 0.0
    assert float(output["stats/density"][0]) > 0.0
    assert output["field"].shape[1] == 2

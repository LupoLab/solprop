# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify tube-lattice collection, translation, propagation and saved-run contracts."""

import importlib.util
import math
from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import pytest

from march_bounds import FIXED_ORDER, assert_matched, simulation_bound
from solprop.boundaries import Absorber
from solprop.constants import C
from solprop.driver import Simulation, prop_capillary
from solprop.driver.material_output import _cross_sections
from solprop.exceptions import DriverError, ModeError
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.modes import MarcatiliMode, ModeCollection, TubeLattice, VincettiMode
from solprop.modes.base import alpha, beta, beta1, zdw
from solprop.modes.quadrature import mode_matrix, transverse_quadrature
from solprop.output import open_file
from solprop.output.hdf5 import resume
from solprop.output.schema import ModeRecord
from solprop.physdata import Component, Mixture, Species, TabulatedSusceptibility
from solprop.profiles import Constant
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LATTICE = TubeLattice(180e-9, 9e-6, 6)
BACKENDS = ["numpy_ref", "numba_cpu"]
if importlib.util.find_spec("solprop_native") is not None:
    BACKENDS.append("native")


# Heterogeneous constructor keywords are kept in this test-only mapping. The grid
# is the smallest that carries the 450-1200 nm band: 64 points over 60 fs; a
# two-mode set asks for the wider window its walk-off check needs.
def keywords(length=0.01, trange=60e-15, **overrides) -> dict[str, Any]:
    grid = EnvelopeGrid.from_limits(length, 800e-9, (450e-9, 1200e-9), trange)
    boundary = Absorber(
        np.zeros_like(grid.omega),
        np.zeros_like(grid.t),
        np.zeros_like(grid.t_fine),
        grid.active_mask,
        length,
    )
    return {
        "lambda0": 800e-9,
        "grid": grid,
        "fwhm": 20e-15,
        "energy": 1e-8,
        "lattice": LATTICE,
        "cladding": 1.45,
        "envelope": True,
        "kerr": True,
        "thg": False,
        "plasma": False,
        "raman": False,
        "n_r": 8,
        "n_save": 3,
        "absorber": boundary,
        "stats": StatsSpec(names=("energy",)),
        "rtol": 1e-7,
    } | overrides


def build(pressure: float | tuple[float, float] = 0.0, length=0.01, **overrides):
    return Simulation.from_capillary(
        15e-6,
        length,
        "Ar",
        pressure,
        **({"noise": None, **keywords(length, **overrides)}),
    )


def linear(simulation: Simulation) -> Simulation:
    """The same physics objects with no response: the linear-only pattern.

    A run with no nonlinearity is refused by every capillary entry point, so a
    linear check builds the `Simulation` directly, as the parent comparison below
    does.
    """
    return Simulation(
        simulation.grid,
        simulation.modes,
        [],
        simulation.initial,
        settings=simulation.settings,
        profiles=simulation.profiles,
        configuration=simulation.configuration,
    )


def test_collection_quadrature_and_refusal():
    modes = ModeCollection.vincetti(15e-6, "Ar", 0.0, 3, lattice=LATTICE, cladding=1.45)
    assert [cast(VincettiMode, mode).m for mode in modes] == [1, 2, 3]
    assert all(isinstance(mode, VincettiMode) for mode in modes)
    rule = transverse_quadrature(modes, n_r=32)
    matrices = mode_matrix(modes, rule)
    # Same Gauss/Bessel quadrature bound as the existing HE_1m collection tests.
    np.testing.assert_allclose(
        matrices.projection @ matrices.synthesis, np.eye(3), rtol=0, atol=1e-9
    )
    with pytest.raises(ModeError, match="at least one"):
        ModeCollection.vincetti(15e-6, count=0, lattice=LATTICE)


@pytest.mark.parametrize(
    "modes, model, message",
    [
        ([VincettiMode(15e-6, lattice=LATTICE, cladding=1.45)], "full", "given modes"),
        (1, "reduced", "reduced"),
        (0, "full", "at least one"),
    ],
)
def test_capillary_lattice_refusals(modes, model, message):
    with pytest.raises(DriverError, match=message):
        build(modes=modes, model=model)


@pytest.mark.parametrize("lattice", [LATTICE, None])
def test_linear_only_run_is_refused_for_every_mode_set(lattice):
    # The refusal does not depend on the mode class: a tube lattice and a
    # capillary are treated alike, and a linear check builds the problem itself.
    with pytest.raises(DriverError, match="no nonlinearity"):
        build(kerr=False, lattice=lattice)


# The gradient case is medium: the translation resolves the operator's density
# dependence on about 170 Chebyshev nodes, each a mode evaluation, and a gentler
# gradient than the run tests' needs fewer of them for the same representation.
@pytest.mark.parametrize(
    "pressure, representation",
    [
        (0.0, "constant"),
        pytest.param((1.0, 1.2), "expansion", marks=pytest.mark.medium),
    ],
)
def test_translation_is_device_ready(pressure, representation):
    if isinstance(pressure, tuple):
        # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
        with nullcontext():
            simulation = linear(build(pressure=pressure))
            problem = simulation.translate()
    else:
        simulation = linear(build(pressure=pressure))
        problem = simulation.translate()
    assert problem.linear.phi_representation == representation
    assert problem.device_ready
    assert isinstance(simulation.modes[0], VincettiMode)


@pytest.mark.medium
@pytest.mark.parametrize("backend", BACKENDS)
def test_linear_propagation_matches_mode_phase_and_power_loss(backend):
    simulation = linear(build(backend=backend))
    output = simulation.run(progress=NullReporter())
    grid, mode = simulation.grid, simulation.modes[0]
    active = grid.active_mask
    omega = grid.omega[active]
    reference = np.array([grid.omega0])
    phase = (
        beta(mode, omega)
        - beta(mode, reference)[0]
        - beta1(mode, reference)[0] * (omega - grid.omega0)
    )
    multiplier = np.exp((1j * phase - alpha(mode, omega) / 2) * grid.length)
    expected = output["field"][0, ..., active] * multiplier[:, None, None]
    actual = output["field"][-1, ..., active]
    # Translation subtracts optical-scale terms. Its rounding allowance is
    # 64 eps * optical wave number * length; the 1e-12 is the exponential's own
    # rounding over the run's steps, well above the 1e-15 of one evaluation.
    bound = 64 * np.finfo(float).eps * np.max(omega) / C * grid.length + 1e-12
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=float(bound * np.max(abs(output["field"][0])))
    )
    np.testing.assert_allclose(
        abs(actual) ** 2,
        abs(output["field"][0, ..., active]) ** 2
        * np.exp(-alpha(mode, omega) * grid.length)[:, None, None],
        rtol=0,
        atol=float(2 * bound * np.max(abs(output["field"][0])) ** 2),
    )


def absorbing_mixture():
    omega = np.array([1e12, 1e18])
    optical = TabulatedSusceptibility(
        omega,
        np.full(2, 1e-30),
        np.full(2, 2e-32),
        (2 * math.pi * C / omega[-1], 2 * math.pi * C / omega[0]),
        citation="Synthetic absorption witness",
    )
    return Mixture.of(
        Component(
            Species("test species", optical, gamma3=1e-52), Constant(1e24), "test"
        )
    )


def test_material_output_callable_guard_accepts_vincetti():
    mixture = absorbing_mixture()
    mode = VincettiMode(
        15e-6,
        mixture=mixture,
        lattice=LATTICE,
        cladding=1.45,
        loss=np.ones_like,
    )
    # material_output.py's first isinstance site must preserve unavailability
    # when a callable replaces total loss, rather than attributing it to a species.
    assert _cross_sections(mixture, [mode], np.array([2e15])) is None


def test_material_output_species_rows_keep_vincetti():
    mixture = absorbing_mixture()
    mode = VincettiMode(
        15e-6, mixture=mixture, lattice=LATTICE, cladding=1.45, loss=0.5
    )
    omega = np.array([2e15, 3e15])
    # Both the guard and the row-comprehension isinstance sites must include the
    # subclass, freeze its density to one and retain the inherited loss scale.
    result = _cross_sections(mixture, [mode], omega)
    assert result is not None
    assert result.shape == (1, 1, 2)
    np.testing.assert_allclose(
        result[0, 0], np.full(2, 0.5 * 2e-32), rtol=16 * np.finfo(float).eps, atol=0
    )


def test_mixture_rebinding_preserves_subclass():
    mixture = absorbing_mixture()
    original = VincettiMode(15e-6, "He", 1.0, lattice=LATTICE, cladding=1.45)
    rebound = original.with_mixture(mixture)
    assert isinstance(rebound, VincettiMode)
    assert rebound.lattice is original.lattice
    assert rebound.mixture is mixture
    assert rebound.gas is None
    omega = np.array([2e15])
    np.testing.assert_array_equal(
        rebound.neff(omega),
        VincettiMode(15e-6, mixture=mixture, lattice=LATTICE, cladding=1.45).neff(
            omega
        ),
    )
    simulation = Simulation.from_capillary(
        15e-6,
        0.01,
        mixture,
        **({"noise": None, **keywords(lattice=None, modes=[original])}),
    )
    assert isinstance(simulation.modes[0], VincettiMode)
    assert simulation.modes[0].mixture is mixture


def test_records_and_distinct_resume_digest():
    simulation = build(modes=2, trange=120e-15)
    assert len(simulation.modes) == 2
    record = ModeRecord.from_mode(simulation.modes[0])
    assert record.kind == "solprop.modes.antiresonant.VincettiMode"
    assert record.effective_area == MarcatiliMode(15e-6, cladding=1.45).aeff()
    lattice_record = record.configuration["lattice"]
    assert isinstance(lattice_record, dict)
    assert lattice_record["wall_thickness"] == LATTICE.wall_thickness
    assert lattice_record["n_tubes"] == 6
    parent_modes = [
        MarcatiliMode(
            15e-6,
            mixture=cast(VincettiMode, mode).mixture,
            m=cast(VincettiMode, mode).m,
            cladding=1.45,
        )
        for mode in simulation.modes
    ]
    parent = Simulation(
        simulation.grid,
        parent_modes,
        simulation.responses,
        simulation.initial,
        settings=simulation.settings,
        profiles=simulation.profiles,
        configuration=simulation.configuration,
    )
    assert (
        simulation.metadata().configuration_hash()
        != parent.metadata().configuration_hash()
    )
    # The run's own record, without a run: the metadata is what a file carries.
    assert all(
        record.kind == "solprop.modes.antiresonant.VincettiMode"
        for record in simulation.metadata().modes
    )
    assert len(simulation.metadata().modes) == 2


class InterruptedRunError(RuntimeError):
    pass


class Interrupt(NullReporter):
    def plane(self, z, summary):
        raise InterruptedRunError("checkpoint saved")


@pytest.mark.medium
def test_argon_kerr_readback_and_resume(tmp_path):
    length = 0.001
    planes = [length / 2, length]
    schedule = FixedSchedule.uniform(0.0, length, length / 8, save_planes=planes)
    settings = keywords(
        length,
        energy=2e-7,
        schedule=schedule,
        save_planes=planes,
        n_save=None,
        n_r=4,
        # No order choice on the fixed schedule: the resumed run continues one march.
        integrator=FIXED_ORDER,
    )
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        whole = prop_capillary(
            15e-6,
            length,
            "Ar",
            1.0,
            **{"noise": None, **settings},
            progress=NullReporter(),
        )
    path = tmp_path / "vincetti.h5"
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with pytest.raises(InterruptedRunError):
        prop_capillary(
            15e-6,
            length,
            "Ar",
            1.0,
            **{"noise": None, **settings},
            output=path,
            progress=Interrupt(),
        )
    continued = resume(path).march is not None
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        prop_capillary(
            15e-6,
            length,
            "Ar",
            1.0,
            **{"noise": None, **settings},
            output=path,
            resume=True,
            progress=NullReporter(),
        )
    with open_file(path) as stored:
        assert stored.metadata is not None
        assert (
            stored.metadata.modes[0].kind == "solprop.modes.antiresonant.VincettiMode"
        )
        np.testing.assert_array_equal(stored["z"], whole["z"])
        # Resume returns to the same fixed schedule. Where the march's state is kept
        # (the host engines) it continues the Adams history and the run is the
        # uninterrupted one's march, within its rounding budget (SEP 25); an engine
        # that starts the history afresh (Kokkos) restarts at order one, a step error
        # on four coarse steps: 1.4e-10 of the peak, measured.
        peak = float(np.max(abs(whole["field"])))
        departure = float(np.max(abs(stored["field"] - whole["field"]))) / peak
        if continued:
            simulation = Simulation.from_capillary(
                15e-6, length, "Ar", 1.0, **{"noise": None, **settings}
            )
            assert_matched(
                stored["field"], whole["field"], simulation_bound(simulation)
            )
        else:
            assert departure < 1e-8
        assert np.all(np.isfinite(stored["stats/energy"]))


def test_tutorial_fibre_has_zero_dispersion_wavelength():
    mode = VincettiMode(15e-6, lattice=LATTICE)
    root = zdw(mode, wavelength_range=(450e-9, 1000e-9))
    assert 450e-9 < root < 1000e-9

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify supplied table modes through the existing capillary driver."""

import warnings
from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from bounds import step_evaluations
from solprop.constants import C
from solprop.driver import Simulation, prop_capillary
from solprop.driver.material_output import _cross_sections
from solprop.exceptions import DriverError, MaterialRangeWarning, ModeError
from solprop.log import NullReporter
from solprop.modes import (
    IndexTable,
    MarcatiliMode,
    ModeCollection,
    TabulatedHollowMode,
    TabulatedMode,
)
from solprop.physdata import Component, Mixture, Species, TabulatedSusceptibility
from solprop.profiles import Constant
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule


def table_mode(
    cls: type[TabulatedHollowMode] | type[TabulatedMode] = TabulatedHollowMode,
    *,
    m=1,
    **keywords,
):
    omega = 2 * np.pi * C / np.linspace(200e-9, 2e-6, 200)
    vacuum = MarcatiliMode(15e-6, m=m, cladding=1.45)
    table = IndexTable(
        omega, vacuum.neff(omega), area=np.full(omega.size, vacuum.aeff())
    )
    return cls(15e-6, table=table, reference_wavelength=650e-9, m=m, **keywords)


def settings(**overrides) -> dict[str, Any]:
    length = 0.001
    return {
        "wavelength": 650e-9,
        "wavelength_limits": (450e-9, 700e-9),
        "trange": 100e-15,
        "fwhm": 20e-15,
        "energy": 1e-8,
        "envelope": True,
        "kerr": True,
        "thg": False,
        "plasma": False,
        "raman": False,
        "n_r": 8,
        "n_save": 2,
        "stats": StatsSpec(names=("energy",)),
        "schedule": FixedSchedule.uniform(0.0, length, length / 64),
        # Sixty-four steps are far beyond the in-stage collar's limit under the
        # default Adams history (0.7 um); split off, it is integrated exactly.
        "absorber_scheme": "split_step",
    } | overrides


@pytest.mark.medium
def test_gradient_runs_with_one_parameter_on_both_backends():
    mode = table_mode()
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        simulation = Simulation.from_capillary(
            15e-6,
            0.001,
            "Ar",
            (1.0, 1.2),
            modes=[mode],
            **({"noise": None, **settings(backend="numpy_ref")}),
        )
        problem = simulation.translate()
        reference = simulation.run(progress=NullReporter())
        compiled = prop_capillary(
            15e-6,
            0.001,
            "Ar",
            (1.0, 1.2),
            modes=[mode],
            **({"noise": None, **settings(backend="numba_cpu")}),
            progress=NullReporter(),
        )
    assert problem.linear.phi_expansion is not None
    assert problem.linear.phi_expansion.parameters == ("density",)
    first, second = reference.fields("fft"), compiled.fields("fft")
    worst = float(np.max(abs(first - second)) / np.max(abs(first)))
    # The deterministic backend bound counts the schedule's 64 steps, the
    # evaluations' weight in each and the transform reduction length grid.n_t, with
    # 64 eps allocated per sample as the channel ladder does (the leading 64).
    # Nonzero guards against reusing one output.
    evaluations = step_evaluations(problem)
    bound = 64 * 64 * evaluations * simulation.grid.n_t * np.finfo(float).eps
    assert 0 < worst < bound
    assert np.all(np.isfinite(second))
    assert reference.metadata is not None
    assert "medium" in reference.metadata.configuration


def test_filled_table_runs_and_gradient_is_refused():
    mode = table_mode(TabulatedMode)
    # A filled table supplies the whole linear response, so the declared gas is
    # never sampled for dispersion and cannot warn about its own fit range.
    with warnings.catch_warnings():
        warnings.simplefilter("error", MaterialRangeWarning)
        output = prop_capillary(
            15e-6,
            0.001,
            "Ar",
            1.0,
            modes=[mode],
            **({"noise": None, **settings()}),
            progress=NullReporter(),
        )
    assert np.all(np.isfinite(output["field"]))
    with pytest.raises(ModeError, match="table holds one fill"):
        Simulation.from_capillary(
            15e-6,
            0.001,
            "Ar",
            (1.0, 1.2),
            modes=[mode],
            **({"noise": None, **settings()}),
        )


def absorbing_mixture():
    omega = np.array([1e12, 1e18])
    susceptibility = TabulatedSusceptibility(
        omega,
        np.full(2, 1e-30),
        np.full(2, 2e-32),
        (2 * np.pi * C / omega[-1], 2 * np.pi * C / omega[0]),
        citation="Synthetic species absorption witness",
    )
    return Mixture.of(
        Component(
            Species("test species", susceptibility, gamma3=1e-52),
            Constant(1e24),
            "test",
        )
    )


@pytest.mark.parametrize("cls", [TabulatedHollowMode, TabulatedMode])
def test_mixture_rebinding_and_species_material_output(cls):
    mode = table_mode(cls)
    mixture = absorbing_mixture()
    rebound = mode.with_mixture(mixture)
    assert isinstance(rebound, (TabulatedMode, TabulatedHollowMode))
    assert isinstance(rebound, cls)
    assert rebound.table is mode.table
    assert rebound.reference_wavelength == mode.reference_wavelength
    omega = np.array([2e15, 3e15])
    result = _cross_sections(mixture, [rebound], omega)
    assert result is not None
    expected = (
        np.zeros((1, 1, 2)) if cls is TabulatedMode else np.full((1, 1, 2), 2e-32)
    )
    # Synthetic susceptibility, density freezing and unit conversion only.
    np.testing.assert_allclose(result, expected, rtol=16 * np.finfo(float).eps, atol=0)
    simulation = Simulation.from_capillary(
        15e-6,
        0.001,
        mixture,
        modes=[mode],
        **({"noise": None, **settings()}),
    )
    metadata = simulation.metadata()
    assert "medium" in metadata.configuration
    callable_mode = table_mode(cls, mixture=mixture, loss=np.ones_like)
    assert _cross_sections(mixture, [callable_mode], omega) is None


def test_mixed_set_uses_existing_orthonormality_guard():
    table = table_mode()
    second = MarcatiliMode(15e-6, m=2, cladding=1.45)
    assert len(ModeCollection((table, second))) == 2
    valid = Simulation.from_capillary(
        15e-6,
        0.001,
        "Ar",
        1.0,
        modes=[table, second],
        **({"noise": None, **settings(trange=200e-15)}),
    )
    assert len(valid.modes) == 2
    # A tabulated HE11 beside an analytic one is the same channel twice, and a
    # collection now refuses that by its label before the overlap is looked at.
    with pytest.raises(ModeError, match="duplicate physical channel"):
        Simulation.from_capillary(
            15e-6,
            0.001,
            "Ar",
            1.0,
            modes=[table, MarcatiliMode(15e-6)],
            **({"noise": None, **settings()}),
        )
    # The orthonormality gate is for a set that is not a repeat: a mode and a
    # partner turned by 0.3 rad are two channels whose overlap is cos(0.3).
    with pytest.raises(DriverError, match="not an orthonormal set"):
        Simulation.from_capillary(
            15e-6,
            0.001,
            "Ar",
            1.0,
            modes=[table, MarcatiliMode(15e-6).rotated(0.3)],
            **({"noise": None, **settings()}),
        )

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Split and two-channel field witnesses with Luna's emitted constants held fixed."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gated_source import _gated_source
from mixture_cases import AnalyticRate
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import EPS0, C
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import (
    Component,
    IonisationData,
    Mixture,
    Species,
    TabulatedSusceptibility,
)
from solprop.profiles import Constant
from solprop.stepper import DOPRI5, FixedSchedule, PIController
from solprop.translate.linop import constant_phi, power_weight_at, tabulate
from solprop.translate.problem import assemble

DATA = json.loads(
    (Path(__file__).parents[1] / "data/luna_mixture_propagation.json").read_text()
)
PARAMETERS = DATA["parameters"]
LUNA = DATA["luna"]


def spectrum(tag):
    return np.asarray(LUNA[tag + "_RE"]) + 1j * np.asarray(LUNA[tag + "_IM"])


def problem_for(tag, steps):
    p = PARAMETERS
    # The witness is frozen (``docs/dev/testing.md``): every input below is stated
    # rather than left to a solprop default -- the grid rules, the mode's model, the
    # integrator, no absorber and no noise, each today's default.
    grid = FieldGrid.from_limits(
        p["length"],
        p["wavelength"],
        (p["lambda_min"], p["lambda_max"]),
        p["trange"],
        ffac=6.0,
        fft_size="pow2",
        collar_fraction=0.05,
    )
    # Luna evaluates dt and the frequency spacing in a different order. Use its
    # emitted axes for this matched-input witness; production grids remain unchanged.
    time, fine_time = np.asarray(LUNA["T"]), np.asarray(LUNA["TO"])
    omega = np.asarray(LUNA["OMEGA"])
    grid = dataclasses.replace(
        grid,
        t=time,
        t_fine=fine_time,
        omega=omega,
        omega_fine=np.arange(grid.n_to // 2 + 1) * omega[1],
        dt=float(time[len(time) // 2 + 1]),
        dt_fine=float(fine_time[len(fine_time) // 2 + 1]),
    )
    grid.validate()
    assert_array_equal(grid.omega, LUNA["OMEGA"])
    optical = TabulatedSusceptibility(
        np.array([1e12, 1e18]),
        np.full(2, 1e-30),
        np.zeros(2),
        (2 * np.pi * C / 1e18, 2 * np.pi * C / 1e12),
        citation="Synthetic optics; the witness supplies the exact linear operator",
    )
    components = []
    responses = []
    for index, (density, gamma3) in enumerate(
        zip(LUNA[tag + "_DENSITY"], LUNA[tag + "_GAMMA3"], strict=True)
    ):
        label = f"population-{index}"
        potential = p["potentials"][index] if tag == "PLASMA" else 1e-18
        material = Species(
            label, optical, gamma3=gamma3, ionisation=IonisationData(potential)
        )
        components.append(Component(material, Constant(density), label))
        responses.append(KerrResponse(gamma3, thg=True, species=label))
        if tag == "PLASMA":
            responses.append(
                PlasmaResponse(
                    material,
                    # Luna's column is the cumulative trapezoid (SEP 18).
                    integration="trapezoid",
                    rate=AnalyticRate(
                        scale=p["rate_scales"][index],
                        reference=p["rate_reference"],
                        threshold=0.0,
                    ),
                    species=label,
                )
            )
    mixture = Mixture.of(*components)
    mode = MarcatiliMode(
        p["radius"], mixture=mixture, cladding=1.45, loss=False, model="full"
    )
    problem = assemble(
        grid,
        mode,
        responses,
        spectrum(tag + "_INITIAL")[None, None, :],
        controller=PIController(rtol=p["rtol"], atol=p["atol"]),
        tableau=DOPRI5,
        absorber=None,
        noise=None,
        save_planes=[grid.length],
        dz_max=grid.length / steps,
        dz_init=grid.length / steps,
        schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / steps),
        ionisation_field_range=(1.0, 1e12) if tag == "PLASMA" else None,
    )
    _, epsilon0, charge_mass_ratio, nlscale = LUNA["CONSTANTS"]
    low, high = problem.linear.z_span
    aeff = LUNA[tag + "_AEFF"][0]
    transform = dataclasses.replace(
        problem.transform,
        synthesis=np.array([[grid.pad_scale / (nlscale * np.sqrt(aeff))]]),
        aeff=tabulate(lambda z: np.full(z.shape, aeff), low, high),
    )
    kerr = problem.responses.kerr_coefficients
    assert kerr is not None
    channels = tuple(
        dataclasses.replace(channel, charge_mass_ratio=charge_mass_ratio)
        for channel in problem.responses.plasma_channels
    )
    prefactor = (
        spectrum(tag + "_PREFACTOR")
        * np.asarray(LUNA["OMEGA_WINDOW"])
        * grid.crop_scale
    )[None, None, :]
    compiled = dataclasses.replace(
        problem.responses,
        kerr_coefficients=dataclasses.replace(
            kerr, coefficient=kerr.coefficient * epsilon0 / EPS0
        ),
        plasma_channels=channels,
        prefactor_constant=prefactor,
        prefactor=tabulate(
            lambda z: np.repeat(prefactor[None], len(z), axis=0), low, high
        ),
        prefactor_expansion=None,
    )
    fine_window = np.asarray(LUNA["TO_WINDOW"])
    fine_window.setflags(write=False)
    problem = dataclasses.replace(
        problem,
        transform=transform,
        responses=compiled,
        linear=dataclasses.replace(
            constant_phi(spectrum(tag + "_LINEAR"), grid.length),
            power_weight_representation=problem.linear.power_weight_representation,
            power_weight=problem.linear.power_weight,
            power_weight_constant=problem.linear.power_weight_constant,
        ),
        # This source-level witness matches Luna's physical electric-field launch.
        # The user launch conversion has separate tests; retain this exact
        # internal amplitude so the archived coefficient witness keeps its scope.
        initial=dataclasses.replace(
            problem.initial,
            field0=spectrum(tag + "_INITIAL")[None, None, None, :],
        ),
        grids=dataclasses.replace(problem.grids, t_window_fine=fine_window),
    )
    problem.validate()
    return problem, grid


@pytest.mark.parametrize("tag", ["SINGLE", "SPLIT"])
def test_luna_mixture_field_witness(tag, backend_class):
    check_luna_mixture_field_witness(tag, backend_class)


def test_luna_plasma_witness_retains_the_historical_gated_equation():
    # The archived array and its tolerance retain their original equation. The
    # production current source has independent all-backend mixture ladders.
    with _gated_source():
        check_luna_mixture_field_witness("PLASMA", NumpyRefBackend)


def check_luna_mixture_field_witness(tag, backend_class):
    steps = 64
    problem, grid = problem_for(tag, steps)
    channels = max(1, problem.flags.n_channels)
    # This coefficient and refinement formula were recorded before Luna ran.
    # The finest DOPRI5 remainder is estimated from the preceding refinement;
    # arithmetic accumulates 64 eps/sample/channel/stage/step.
    rounding = (
        DATA["bounds"]["roundoff_coefficient"]
        * grid.n_to
        * channels
        * 7
        * steps
        * np.finfo(float).eps
    )
    # Convert both stored Luna amplitudes with solprop's mode index at the plane.
    # Archived witness bytes and its predeclared error budget remain unchanged.
    root = np.sqrt(power_weight_at(problem, grid.length)).reshape(-1)
    reference = spectrum(f"{tag}_FINAL_{steps}") * root
    refinement = np.linalg.norm(reference - spectrum(f"{tag}_FINAL_32") * root) / 31
    bound = rounding + 2 * refinement / np.linalg.norm(reference)
    assert bound > 0.0
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(grid.length)
        assert summary.n_steps == steps == int(LUNA[f"{tag}_RUN_64"][0])
        actual = backend.observe().require_field().reshape(-1) * root
    finally:
        backend.close()
    discrepancy = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    assert discrepancy < bound
    # The finest single-gas refinements reach arithmetic noise rather than an
    # asymptotic fifth-order sequence. Agreement also meets the arithmetic part
    # alone, so acceptance does not rely on that refinement estimate.
    assert discrepancy < rounding


def test_luna_split_witness_has_the_predeclared_summation_bound():
    for steps in PARAMETERS["steps"]:
        bound = 64 * len(LUNA["TO"]) * 2 * 7 * steps * np.finfo(float).eps
        assert bound > 0.0
        whole, split = (
            spectrum(f"SINGLE_FINAL_{steps}"),
            spectrum(f"SPLIT_FINAL_{steps}"),
        )
        assert np.linalg.norm(whole - split) / np.linalg.norm(whole) < bound

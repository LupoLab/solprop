# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's two demonstration cases as short runs of the dispersive policies.

The argon deep-ultraviolet four-wave mixing case (a 400 nm pump and an 800 nm seed
making 266.7 nm at the linear phase-matching pressure) and the helium RDW case of
milestone M2 (its geometry, grid and pulse; the reduced, two-mode size), each over ten
fixed steps on ``numpy_ref`` under all three electronic policies. The runs complete
and are finite; their manifests record the regions SEP 6, section 4.10, assigns their
bands; with the dispersion coefficients zeroed and the static value set to the
carrier's, each dispersive policy reproduces the carrier run to rounding; and on a
near-infrared-only window the symmetric factorisation stays within the bound its
coefficient error sets on the nonlinear change of the field. The full-length cases are
the tutorial's (``docs/tutorials/capillaries/plot_11_dispersive_kerr.py``).
"""

import dataclasses
import math
import warnings
from typing import Any

import numpy as np
import pytest

from solprop.driver import prop_capillary
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import FieldGrid
from solprop.physdata import MaterialSelection
from solprop.physdata.nonlinear_data import DispersiveKerrModel, OptmatKerr
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule

POLICIES = ("carrier-kerr", "symmetric-kerr", "gas-polynomial")
DISPERSIVE = POLICIES[1:]
STEPS = 10

ARGON_PRESSURE = 0.2094
"""The linear phase-matching pressure of 400 + 400 - 800 -> 266.7 nm in the HE_11 mode
of a 50 um-radius argon capillary, bar: the zero of ``2 beta_p - beta_i - beta_s`` from
``MarcatiliMode.neff_minus_one``, as the tutorial finds it (209.4 mbar)."""

CASES: dict[str, dict[str, Any]] = {
    # The argon DUV FWM case: 210 nm keeps the active band, to 1.1 omega_max (190.9 nm),
    # inside argon's 189.4 nm band edge.
    "argon": {
        "arguments": (50e-6, 2e-3, "Ar", ARGON_PRESSURE),
        "wavelength": 400e-9,
        "pulses": [
            GaussianPulse(wavelength=400e-9, fwhm=30e-15, energy=10e-6),
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=10e-6),
        ],
        "wavelength_limits": (210e-9, 2e-6),
        "trange": 250e-15,
    },
    # An 800 nm pulse in argon on a near-infrared window, 545 nm to 2.4 um: 2 cm at
    # 1 bar, about 0.2 rad of nonlinear phase.
    "near-infrared": {
        "arguments": (50e-6, 2e-2, "Ar", 1.0),
        "wavelength": 800e-9,
        "fwhm": 30e-15,
        "energy": 20e-6,
        "plasma": False,
        "wavelength_limits": (600e-9, 1.2e-6),
        "trange": 250e-15,
    },
    # Milestone M2's geometry, grid and pulse at its entrance pressure: the reduced
    # size (two modes), over its first 2 mm.
    "helium": {
        "arguments": (125e-6, 2e-3, "He", 1.2),
        "wavelength": 800e-9,
        "fwhm": 7.5e-15,
        "energy": 275e-6,
        "modes": 2,
        "wavelength_limits": (100e-9, 1.5e-6),
        "trange": 200e-15,
    },
}

DEMONSTRATIONS = ("argon", "helium")
"""SEP 6's two demonstration cases; the third is the near-infrared window."""


def run(case: str, policy: str, **overrides: Any):
    """Return ten fixed steps of a case under an electronic policy.

    The default collar of a guide this short is too stiff for an explicit step of a
    tenth of it, so it is integrated exactly by the split-step absorber scheme.
    """
    keywords = dict(CASES[case] | overrides)
    radius, length, gas, pressure = keywords.pop("arguments")
    grid = FieldGrid.from_limits(
        length,
        keywords["wavelength"],
        keywords.pop("wavelength_limits"),
        keywords.pop("trange"),
    )
    keywords |= {
        "grid": grid,
        "absorber_scheme": "split_step",
        "schedule": FixedSchedule.uniform(0.0, length, length / STEPS),
        "n_save": 2,
        "backend": "numpy_ref",
        "materials": MaterialSelection(electronic=policy),
        "progress": "none",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return prop_capillary(
            radius, length, gas, pressure, **{"noise": None, **keywords}
        )


def exit_field(output) -> np.ndarray:
    """Return the field of the last saved plane."""
    return np.asarray(output["field"])[-1]


def regions_of(output) -> Any:
    """Return the regions of a one-gas run's dispersive response, from its manifest."""
    (entry,) = output.metadata.materials.electronic
    return entry["regions"]


@pytest.fixture(scope="module")
def runs():
    """Return every case under every policy, run once for the module."""
    return {
        (case, policy): run(case, policy)
        for case in DEMONSTRATIONS
        for policy in POLICIES
    }


@pytest.mark.medium
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("case", DEMONSTRATIONS)
def test_the_runs_complete_and_are_finite(runs, case, policy):
    output = runs[case, policy]
    field = exit_field(output)
    assert np.all(np.isfinite(field))
    assert np.asarray(output["z"])[-1] == pytest.approx(CASES[case]["arguments"][1])
    energies = np.asarray(output["stats/energy"])
    assert np.all(np.isfinite(energies)) and np.sum(energies[-1]) > 0.0
    assert output.metadata.backend.name == "numpy_ref"


@pytest.mark.medium
@pytest.mark.parametrize("policy", DISPERSIVE)
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("argon", {"supported", "bounded"}),
        ("helium", {"supported", "bounded", "best_effort"}),
    ],
)
def test_the_regions_are_recorded(runs, case, policy, expected):
    """Argon's band stays inside its band edge; helium's reaches below 120.3 nm."""
    regions = regions_of(runs[case, policy])
    materials = runs[case, policy].metadata.materials
    present = {
        name for name in ("supported", "bounded", "best_effort") if regions[name]
    }
    assert present == expected
    law: Any = materials.policies["electronic_response"]
    assert law["law"] == policy
    model = OptmatKerr(CASES[case]["arguments"][2]).dispersive_model()
    assert regions["supported"][0] == pytest.approx(1.0 / model.leg_bound, rel=1e-12)
    if case == "helium":
        assert model.band_edge_linear is not None
        assert regions["best_effort"][1] == pytest.approx(
            1.0 / model.band_edge_linear, rel=1e-12
        )


def flat_model(gas: str, carrier: float) -> DispersiveKerrModel:
    """Return a gas's model without dispersion, its static value the carrier's."""
    model = OptmatKerr(gas).dispersive_model()
    return dataclasses.replace(
        model, gamma3_static=OptmatKerr(gas).gamma3(carrier), a=0.0, b=0.0, b_prime=0.0
    )


@pytest.mark.medium
@pytest.mark.parametrize("policy", DISPERSIVE)
@pytest.mark.parametrize("case", DEMONSTRATIONS)
def test_without_dispersion_a_dispersive_run_is_the_carrier_run(
    runs, monkeypatch, case, policy
):
    """The whole driver path, with the material layer's model replaced.

    With ``a = b = b' = 0`` and the static coefficient the carrier's Kerr value, both
    evaluators' full cubic is the constant cubic: the symmetric filter is one and the
    polynomial's extra products are weighted by zero. What is left is the leaf's extra
    transform round trips in every stage, each a few ``eps log2(n_to)`` of the
    polarisation, which is itself a small part of the field's change over 2 mm: the
    exit fields agree to 1e-13 of their largest bin (bound; measured 4e-16 to 8e-16),
    fixed steps keeping the trajectories identical. The real models move them by 4e-4
    to 1e-3.
    """
    gas = CASES[case]["arguments"][2]
    flat = flat_model(gas, CASES[case]["wavelength"])
    monkeypatch.setattr(OptmatKerr, "dispersive_model", lambda self: flat)
    got = exit_field(run(case, policy))
    expected = exit_field(runs[case, "carrier-kerr"])
    scale = float(np.abs(expected).max())
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13 * scale)
    # The comparison has teeth: the real model moves the field far beyond rounding.
    real = exit_field(runs[case, policy])
    assert np.abs(real - expected).max() > 1e-5 * scale


def symmetric_error_bound(
    model: DispersiveKerrModel, band: tuple[float, float]
) -> float:
    """Return the largest symmetric-model error over quartets in a band (SEP 6, 4.7).

    SEP 6's survey procedure on the run's active band: ``default_rng(4)`` draws three
    input wavelengths uniformly over the band, the output ``nu_1 + nu_2 - nu_3`` is
    kept when it lies in the band too, and each quartet's bounded symmetric coefficient
    is compared with the bounded polynomial.
    """
    shortest, longest = band
    legs = 1.0 / np.random.default_rng(4).uniform(shortest, longest, (200_000, 3))
    output = legs[:, 0] + legs[:, 1] - legs[:, 2]
    kept = (output > 1.0 / longest) & (output < 1.0 / shortest)
    quartets = np.column_stack([output[kept], legs[kept]])
    polynomial = model.gamma3(quartets) / model.gamma3_static
    kerr = model.kerr_ratio(np.minimum(quartets**2, model.leg_bound**2))
    symmetric = np.prod(kerr**0.25, axis=-1)
    return float(np.max(np.abs(symmetric / polynomial - 1.0)))


@pytest.mark.medium
def test_on_a_near_infrared_window_the_evaluators_differ_within_the_bound():
    """Symmetric against polynomial on an 800 nm pulse in argon, 545 nm to 2.4 um.

    To first order in the nonlinearity the change of the field over a short window is
    the integrated polarisation, so the evaluators' relative difference in that change
    is a weighted mean of their quartets' relative coefficient difference, below its
    maximum over the band. 2 cm of 1 bar argon, a 20 uJ 30 fs pulse: about 0.2 rad of
    nonlinear phase. The bound is the band's sampled maximum, 0.34 %; the measured
    difference, 6.3e-6, is far below it because self-phase modulation, exact in both,
    dominates.
    """
    # The linear propagation, from a launch a million times weaker scaled back up: its
    # nonlinear residue is a part in 1e6 of the change measured.
    weak = exit_field(run("near-infrared", "carrier-kerr", energy=20e-12))
    linear = weak * math.sqrt(20e-6 / 20e-12)
    symmetric_run = run("near-infrared", "symmetric-kerr")
    symmetric = exit_field(symmetric_run) - linear
    polynomial = exit_field(run("near-infrared", "gas-polynomial")) - linear
    regions = regions_of(symmetric_run)
    band = tuple(regions["band"])
    assert regions["bounded"] is None and regions["best_effort"] is None
    bound = symmetric_error_bound(OptmatKerr("Ar").dispersive_model(), band)
    assert 0.0 < bound < 0.00913  # within argon's 400-2000 nm survey maximum
    difference = np.linalg.norm(symmetric - polynomial) / np.linalg.norm(polynomial)
    assert 0.0 < difference < bound
    # The window has a nonlinear phase worth measuring: the change is not rounding.
    assert np.linalg.norm(polynomial) > 1e-3 * np.linalg.norm(linear)

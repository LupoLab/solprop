# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression of the Vincetti model against the reference implementation's FEM rows.

``tests/data/jcmwave_vincetti.json`` holds JCMsuite finite-element solutions of the
fundamental mode of a 30 um core, 180 nm wall, six-tube fibre from pyJCMwave's
library, evacuated and with argon at 10 bar and 293 K, with the run's own silica and
argon indices. The fibre's t/r = 0.017 lies outside the fits' validated 0.05-0.2, so
every bound below is the residual measured when the model was validated, with a 15 %
margin: a regression guard on this one fibre, not an accuracy claim. The sample
weighting is by the stored wavelengths, and the near/away split is a descriptive 3 %
distance from the vacuum wall resonances (``docs/guide/waveguides/modes.md``, "Accuracy
against a
finite-element solution").
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Complex, Float
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from witness_materials import kingston_argon

from solprop.constants import C
from solprop.modes import TubeLattice, VincettiMode
from solprop.modes.base import beta2, db_per_m
from solprop.physdata.glasses import GlassIndex

WITNESS = json.loads(
    (Path(__file__).parents[1] / "data" / "jcmwave_vincetti.json").read_text()
)
GEOMETRY = WITNESS["geometry"]
RESONANCES = np.asarray(WITNESS["resonance_wavelengths_m"])
LATTICE = TubeLattice(
    GEOMETRY["wall_thickness_m"], GEOMETRY["tube_radius_m"], GEOMETRY["n_tubes"]
)
# The report's split: a row within 3 % of a vacuum wall resonance is "near".
NEAR_FRACTION = 0.03
# The nine smooth argon rows at 690-946 nm used for the gas differential, and the
# vacuum rows the differential interpolates between (500 nm and above, cubic).
DIFFERENTIAL_ARGON_ABOVE = 650e-9
DIFFERENTIAL_VACUUM_ABOVE = 500e-9
# The interior of the first transmission window, where the loss fit is compared as
# a ratio; the band edges are compared through the logarithmic RMSE instead.
WINDOW_INTERIOR = (480e-9, 1150e-9)
FIRST_BAND = (580e-9, 1150e-9)


@dataclass(frozen=True, kw_only=True, eq=False)
class StoredGlass(GlassIndex):
    """The FEM run's silica lookup, splined through its own sample points."""

    spline: CubicSpline

    def _index_and_derivative(
        self, omega: Float[np.ndarray, "*shape"]
    ) -> tuple[Complex[np.ndarray, "*shape"], Complex[np.ndarray, "*shape"]]:
        # An interpolating spline returns the table value exactly at a knot, which
        # is where the index comparisons evaluate; its derivative serves the beta_2
        # differences, whose relative step stays between the knots.
        values = np.asarray(self.spline(omega))
        slope = np.asarray(self.spline(omega, 1))
        return values.astype(complex), slope.astype(complex)


@dataclass(frozen=True)
class Evaluation:
    wavelength: np.ndarray
    fem_index: np.ndarray
    fem_loss: np.ndarray
    fem_area: np.ndarray
    gas_index: np.ndarray
    index: np.ndarray
    loss: np.ndarray
    area: np.ndarray
    mode: VincettiMode

    @property
    def away(self):
        distance = np.min(
            np.abs(self.wavelength[:, None] / RESONANCES[None, :] - 1), axis=1
        )
        return distance >= NEAR_FRACTION

    @property
    def band(self):
        return np.sum(self.wavelength[:, None] < RESONANCES[None, :], axis=1)

    @property
    def residual(self):
        return self.index - self.fem_index

    def rmse(self, mask):
        return float(np.sqrt(np.mean(self.residual[mask] ** 2)))


def stored_glass(rows):
    wavelength, index = rows[:, 0], rows[:, 2]
    omega = 2 * math.pi * C / wavelength
    order = np.argsort(omega)
    # The range is widened by the relative step of `beta2`, whose differences
    # reach just beyond the outermost knots.
    return StoredGlass(
        name="SiO2",
        source="pyjcmwave lookup",
        wavelength_range=(
            0.99 * float(wavelength.min()),
            1.01 * float(wavelength.max()),
        ),
        citation=WITNESS["provenance"]["source_sha256"]["pyjcmwave/data/silica_n.csv"],
        spline=CubicSpline(omega[order], index[order]),
    )


def evaluate(fill, *, evacuated=False):
    rows = np.asarray(fill["rows"])
    wavelength = rows[:, 0]
    omega = 2 * math.pi * C / wavelength
    if fill["gas"] is None or evacuated:
        mode = VincettiMode(
            GEOMETRY["core_radius_m"], lattice=LATTICE, cladding=stored_glass(rows)
        )
    else:
        # This fit and state reproduce pyJCMwave's argon index to rounding; the
        # assertion below keeps that claim. The fit is the Kingston argon of
        # solprop's former registry, supplied explicitly since SEP 3's Phase 4b.
        mode = VincettiMode(
            GEOMETRY["core_radius_m"],
            fill["gas"],
            fill["pressure_bar"],
            temperature=fill["temperature_k"],
            susceptibility=kingston_argon(),
            lattice=LATTICE,
            cladding=stored_glass(rows),
        )
    terms = mode.sampled_terms(omega)
    density = float(mode.density(np.array([0.0]))[0])
    return Evaluation(
        wavelength,
        rows[:, 3],
        rows[:, 5],
        rows[:, 6],
        np.sqrt(1 + terms.core_susceptibility * density),
        np.real(mode.neff(omega)),
        db_per_m(mode, omega),
        mode.effective_area(omega),
        mode,
    )


@pytest.fixture(scope="module")
def vacuum():
    return evaluate(WITNESS["fills"]["vacuum"])


@pytest.fixture(scope="module")
def argon():
    return evaluate(WITNESS["fills"]["argon"])


def test_materials_match_the_reference_run(vacuum, argon, caplog):
    # The run's own silica table is returned at its knots; its argon index is the
    # same three-pole fit and ideal-gas density as pyJCMwave's.
    for evaluation, fill in ((vacuum, "vacuum"), (argon, "argon")):
        rows = np.asarray(WITNESS["fills"][fill]["rows"])
        # The table returned at its knots, through a wavelength-frequency round
        # trip of a few roundings.
        np.testing.assert_allclose(
            evaluation.mode.sampled_terms(2 * math.pi * C / rows[:, 0]).tube_index,
            rows[:, 2],
            rtol=4 * np.finfo(float).eps,
            atol=0,
        )
        np.testing.assert_allclose(
            evaluation.gas_index, rows[:, 1], rtol=4 * np.finfo(float).eps, atol=0
        )
    # The witness is self-consistent: its loss column is 40 pi Im(n) / (lambda ln 10)
    # rounded to the ten digits of the CSV (measured 3.15e-10 and 3.86e-10).
    for fill in ("vacuum", "argon"):
        rows = np.asarray(WITNESS["fills"][fill]["rows"])
        implied = 40 * math.pi / (rows[:, 0] * math.log(10)) * rows[:, 4]
        np.testing.assert_allclose(implied, rows[:, 5], rtol=4e-10, atol=0)
    with caplog.at_level(logging.WARNING):
        evaluate(WITNESS["fills"]["vacuum"])
    assert "wall_thickness/tube_radius=0.017341 outside [0.05, 0.2]" in caplog.text


def test_vacuum_index_within_measured_residuals(vacuum):
    away = vacuum.away
    # All 79 rows: RMSE 7.528e-6 measured; the lower bound catches a
    # degenerate comparison such as the model compared with itself.
    assert 3.7e-6 < vacuum.rmse(np.ones_like(away)) < 8.7e-6
    # 59 away rows: RMSE 7.449e-6; the residual relative to the FEM's own waveguide
    # decrement has median 2.93 %, p90 11.7 % and max 25.6 %.
    relative = np.abs(vacuum.residual / (vacuum.gas_index - vacuum.fem_index))[away]
    assert vacuum.rmse(away) < 8.6e-6
    assert float(np.median(relative)) < 0.034
    assert float(np.quantile(relative, 0.9)) < 0.135
    assert float(np.max(relative)) < 0.295
    # 20 near rows: RMSE 7.758e-6; the narrower printed dispersion width 0.003 would
    # give 2.5e-5 here.
    assert vacuum.rmse(~away) < 8.9e-6


def test_argon_index_within_measured_residuals(argon):
    away = argon.away
    # 14 away rows: RMSE 4.452e-6 and median 0.71 % of the decrement.
    relative = np.abs(argon.residual / (argon.gas_index - argon.fem_index))[away]
    assert 2.2e-6 < argon.rmse(away) < 5.1e-6
    assert float(np.median(relative)) < 0.0082


def test_gas_differential_within_measured_residual(vacuum, argon):
    # The change of the FEM index from vacuum to 10 bar, against the model's own
    # change at the same wavelengths, removes the fixed vacuum-model bias. Nine
    # argon rows at 690-946 nm; the vacuum FEM refractivity is a cubic spline over
    # its rows at 500 nm and above (a quartic agrees to 1e-16). Measured 7.22e-8
    # with the single-power decrement, 4.60e-7 with the legacy 1/n_g^2.
    keep = argon.wavelength > DIFFERENTIAL_ARGON_ABOVE
    assert int(keep.sum()) == 9
    vacuum_rows = vacuum.wavelength >= DIFFERENTIAL_VACUUM_ABOVE
    fem_vacuum = 1 + InterpolatedUnivariateSpline(
        vacuum.wavelength[vacuum_rows], vacuum.fem_index[vacuum_rows] - 1, k=3
    )(argon.wavelength[keep])
    model_vacuum = evaluate(WITNESS["fills"]["argon"], evacuated=True).index
    error = (argon.index[keep] - model_vacuum[keep]) - (
        argon.fem_index[keep] - fem_vacuum
    )
    assert 3.6e-8 < float(np.max(np.abs(error))) < 8.3e-8
    # The FEM area itself is fill independent to 1e-4 at these rows (n_g^2 - 1 is
    # 5.3e-3), which is what the area's 1/n_g prefactor preserves.
    fem_vacuum_area = InterpolatedUnivariateSpline(
        vacuum.wavelength[vacuum_rows], vacuum.fem_area[vacuum_rows], k=3
    )(argon.wavelength[keep])
    assert float(np.max(np.abs(argon.fem_area[keep] / fem_vacuum_area - 1))) < 1.2e-4


def test_loss_within_measured_ratios(vacuum, argon):
    for evaluation in (vacuum, argon):
        away = evaluation.away
        ratio = evaluation.fem_loss / evaluation.loss
        # Over the interior of the first window (480-1150 nm) the FEM loss is 0.5 to
        # 1.7 times the fit's: measured 0.526-1.61 in vacuum (22 rows) and
        # 0.70-1.03 in argon (9 rows).
        interior = away & (evaluation.wavelength > WINDOW_INTERIOR[0])
        interior &= evaluation.wavelength < WINDOW_INTERIOR[1]
        assert np.all(ratio[interior] > 0.5 / 1.15)
        assert np.all(ratio[interior] < 1.7 * 1.15)
    # Over all 59 vacuum away rows, band edges included, the fit's Lorentzians are
    # far narrower than the FEM resonances: FEM/model spans 0.29-41 and the
    # logarithmic RMSE is 0.383.
    ratio = (vacuum.fem_loss / vacuum.loss)[vacuum.away]
    assert float(np.sqrt(np.mean(np.log10(ratio) ** 2))) < 0.44
    assert np.all(ratio > 0.25)
    assert np.all(ratio < 47.5)


def test_area_within_measured_residuals(vacuum, argon):
    # Away rows: the fitted area's median residual is 2.48 % in vacuum and 1.16 % in
    # argon, its maximum 27.9 % / 26.9 % at the band edges.
    for evaluation, median, worst in ((vacuum, 0.0285, 0.32), (argon, 0.0134, 0.31)):
        away = evaluation.away
        residual = np.abs(evaluation.area[away] / evaluation.fem_area[away] - 1)
        assert float(np.median(residual)) < median
        assert float(np.max(residual)) < worst


def test_first_band_beta2_discrepancy_is_recorded(vacuum):
    # The FEM's own beta_2 in the first band, three-point second differences of
    # beta = omega n / c on the raw rows, is more negative than the model's by 1 %
    # at 607 nm, 18 % at 782 nm, 48 % at 913 nm and 70 % at 1013 nm (-1268 against
    # -744 fs^2/m): the curvature of the fit's U-shaped index residual on this
    # t/r = 0.017 fibre. Measured worst 0.703; the lower bound records the
    # finding, so a smaller number is an improvement to record there, not a
    # regression.
    omega = 2 * math.pi * C / vacuum.wavelength
    model = beta2(vacuum.mode, omega)
    beta = omega * vacuum.fem_index / C
    rows = np.where(
        (vacuum.wavelength > FIRST_BAND[0]) & (vacuum.wavelength < FIRST_BAND[1])
    )[0]
    worst = 0.0
    for row in rows[1:-1]:
        below, above = omega[row] - omega[row + 1], omega[row - 1] - omega[row]
        fem = (
            2
            * (
                beta[row - 1] * below
                - beta[row] * (below + above)
                + beta[row + 1] * above
            )
            / (below * above * (below + above))
        )
        worst = max(worst, abs(fem / model[row] - 1))
    assert 0.35 < worst < 0.82

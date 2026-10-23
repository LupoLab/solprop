# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The band rule on ionising fields, against an independent reference (SEP 18, item 4).

Section 7 item 4's protocol, complete. One scalar column of the field-grid plasma on the
default grid (``ffac = 6``) under the band rule, against a Richardson extrapolation of
the plain trapezoid column at ``ffac = 48`` and ``96``, ``R = (4 P96 - P48) / 3``, whose
uncertainty is the same extrapolation from ``24`` and ``48``. The reference shares no
filter, floor or spectral antiderivative with the rule it qualifies (section 4.8).

The quantity is the polarisation's coarse spectrum ``FFT(P w_fine) dt``; the error in
each band of ``omega / omega_max`` is the relative L2 error of that complex spectrum,
asserted where the band holds at least ``1e-6`` of the active band's ``|P|^2`` and
recorded otherwise. The active band's error, normalised by its own norm, is asserted on
every field.
"""

import numpy as np
import pytest

from convergence_records import Column, Study, record
from solprop.backends.numpy_ref import passes
from solprop.constants import C
from solprop.grid import FieldGrid
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse
from solprop.translate.plasma_integration import design_running_integral
from solprop.translate.responses import plasma_coefficients

DENSITY = 2.0e25
"""Number density of section 7 item 4, m^-3."""

HELIUM_LIMITS = (90e-9, 4e-6)
ARGON_LIMITS = (110e-9, 4e-6)
PUMP = 800e-9
SHORT_SEEDS = {
    "He": ((120e-9, 4e-15, 0.03), (100e-9, 4e-15, 0.03), (93e-9, 4e-15, 0.03)),
    "Ar": ((160e-9, 4e-15, 0.03), (125e-9, 4e-15, 0.03), (113e-9, 4e-15, 0.03)),
}

CASES = {
    # name: gas, limits, peak field in V/m, pump FWHM in s, seeds (wavelength, FWHM,
    # relative amplitude)
    "He 6e10 V/m, 1.3 fs": ("He", HELIUM_LIMITS, 6e10, 1.3e-15, SHORT_SEEDS["He"]),
    "He 9e10 V/m, 1.3 fs": ("He", HELIUM_LIMITS, 9e10, 1.3e-15, SHORT_SEEDS["He"]),
    "Ar 4e10 V/m, 1.3 fs": ("Ar", ARGON_LIMITS, 4e10, 1.3e-15, SHORT_SEEDS["Ar"]),
    "Ar 8e10 V/m, 1.3 fs (saturating)": (
        "Ar",
        ARGON_LIMITS,
        8e10,
        1.3e-15,
        SHORT_SEEDS["Ar"],
    ),
    "Ar 2.5e10 V/m, 30 fs": (
        "Ar",
        ARGON_LIMITS,
        2.5e10,
        30e-15,
        ((125e-9, 10e-15, 0.03),),
    ),
}

BANDS = ((0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0), (1.0, 1.1))
"""The bands of omega / omega_max; the last is the collar, its top included."""

FLOOR = 1e-6
"""A band's share of the active ``|P|^2`` below which its error is recorded, not
asserted (accepted with the SEP)."""

BAND_TARGET = 1e-4
ACTIVE_TARGET = 1e-5
WORK_DEFECT_TARGET = 1e-6
WORK_TARGET = 1e-5
"""Section 7 item 3: the source's work through the transform against int E J."""

PLASMA_BAND_RESPONSE = Study(
    name="plasma_band_response",
    title="The band rule's plasma polarisation on ionising fields, against Richardson",
    case={
        "grid": "FieldGrid.from_limits(1.0, 800e-9, limits, 400e-15, ffac) with limits "
        "90 nm - 4 um (He) or 110 nm - 4 um (Ar); ffac 6 for the rule, 24, 48 and 96 "
        "for the reference",
        "field": "transform-limited Gaussians in the coarse spectrum (an 800 nm pump "
        "of amplitude 1 and the seeds of relative amplitude 0.03), zero phase, "
        "centred, times the spectral window, zero-padded onto each fine grid and "
        "normalised to the stated peak on the ffac 6 grid",
        "medium": "optmat's default rate, OptmatRate(gas, 800e-9), tabulated over "
        "[1e8, 3 E_peak] V/m; density 2e25 m^-3; no pre-ionisation; one scalar "
        "column",
        "reference": "R = (4 P96 - P48) / 3 of the plain trapezoid column; its "
        "uncertainty (4 P48 - P24) / 3 against R",
        "norm": "relative L2 error of FFT(P w_fine) dt in each band of "
        "omega/omega_max, and over the active band",
        "work identity": "the source's work through the transform, "
        "sum E* (i omega) (w P)^ on the kept bins, against int E (w J + w' P) dt "
        "with the collar's spectral derivative (section 4.8), relative",
    },
    axis={"name": "band", "unit": "omega/omega_max", "label": "band"},
    columns=(
        Column(
            name="band_rule",
            unit="1 (relative)",
            why="the qualified quantity; a difference of two spectra each known to "
            "round-off, so stable to far better than the tolerance",
            bound=BAND_TARGET,
            rtol=1e-3,
            atol=1e-12,
        ),
        Column(
            name="trapezoid",
            unit="1 (relative)",
            why="today's rule on the same grid, for scale",
            rtol=1e-6,
            atol=1e-12,
        ),
        Column(
            name="reference",
            unit="1 (relative)",
            why="the reference's own uncertainty, from the third refinement",
            rtol=1e-3,
            atol=1e-12,
        ),
        Column(
            name="share",
            unit="1",
            why="the band's share of the active |P|^2; asserted bands hold >= 1e-6",
            rtol=1e-6,
            atol=1e-300,
        ),
    ),
)


def _grid(limits, ffac):
    return FieldGrid.from_limits(1.0, PUMP, limits, 400e-15, ffac=ffac)


def _field_on(grid, base, spectrum):
    full = np.zeros(grid.n_to // 2 + 1, complex)
    full[: base.n_omega] = spectrum
    return np.fft.irfft(full, grid.n_to) * (grid.n_to / base.n_to)


def _column(field, coefficients, dt, integral):
    """Return the polarisation and the current it integrates, one scalar column."""
    nodes = field[None, None, :]
    rate = passes.ionisation_rate(nodes, coefficients.rate)
    fraction, free = passes.plasma_scan(
        nodes, rate, coefficients, preionisation=0.0, dt=dt, integral=integral
    )
    polarisation = passes.plasma_current(
        nodes,
        rate,
        fraction,
        free,
        coefficients,
        density=DENSITY,
        dt=dt,
        integral=integral,
    )
    squared = field**2
    scale = np.zeros_like(squared)
    np.divide(
        coefficients.ionisation_potential * rate[0] * (1.0 - fraction[0]),
        squared,
        out=scale,
        where=(rate[0] > 0.0) & (squared > 0.0),
    )
    current = DENSITY * (free[0, 0] + scale * field)
    return polarisation[0, 0], current


def work_identity(grid, field, polarisation, current):
    """Return the source's work through the transform against int E (w J + w' P).

    Section 4.8: the band-limited source does work ``sum E* (i omega) (w P)^`` on the
    kept bins, and ``d(wP)/dt = w J + w' P``, the collar's derivative included.
    The field is band-limited, so the time integral is the same sum over every bin.
    Under the band rule ``i omega P^ = H J^`` on the kept bins, so the two agree to
    the multiplier tolerance; under the trapezoid they differ by ``1 - rho``.
    """
    n_to, dt = grid.n_to, grid.dt_fine
    collar = grid.t_window_fine
    omega = 2 * np.pi * np.fft.rfftfreq(n_to, dt)
    slope = np.fft.irfft(1j * omega * np.fft.rfft(collar), n_to)
    kept = np.zeros(n_to // 2 + 1, dtype=bool)
    kept[: grid.n_omega] = grid.active_mask
    weights = np.full(n_to // 2 + 1, 2.0)
    weights[0] = weights[-1] = 1.0
    through = (
        dt
        / n_to
        * np.sum(
            (
                weights
                * np.real(
                    np.conj(np.fft.rfft(field))
                    * 1j
                    * omega
                    * np.fft.rfft(collar * polarisation)
                )
            )[kept]
        )
    )
    direct = dt * np.sum(field * (collar * current + slope * polarisation))
    return abs(through / direct - 1.0)


def study(case):
    """Return the per-band errors of the band rule, the trapezoid and the reference."""
    gas, limits, peak, fwhm, seeds = CASES[case]
    base = _grid(limits, 6.0)
    omega = base.omega

    def gaussian(centre, width, amplitude):
        return amplitude * np.exp(-((omega - centre) ** 2) * width**2 / (8 * np.log(2)))

    spectrum = gaussian(2 * np.pi * C / PUMP, fwhm, 1.0)
    for wavelength, width, amplitude in seeds:
        spectrum = spectrum + gaussian(2 * np.pi * C / wavelength, width, amplitude)
    spectrum = spectrum * base.omega_window * (-1.0) ** np.arange(base.n_omega)
    coefficients = plasma_coefficients(
        PlasmaResponse(gas, OptmatRate(gas, PUMP)), (1e8, 3 * peak), z_max=1.0
    )
    spectra = {}
    scale = None
    diagnostics = None
    work = {}
    for ffac in (6.0, 24.0, 48.0, 96.0):
        grid = _grid(limits, ffac)
        assert np.array_equal(grid.omega[: base.n_omega], omega)
        field = _field_on(grid, base, spectrum)
        scale = scale or peak / float(np.max(np.abs(field)))
        field = field * scale
        rules = {"trapezoid": RunningIntegral.trapezoid()}
        if ffac == 6.0:
            rules["band"] = design_running_integral(grid, "band")
        for name, rule in rules.items():
            polarisation, current = _column(field, coefficients, grid.dt_fine, rule)
            if ffac == 6.0:
                work[name] = work_identity(grid, field, polarisation, current)
            spectra[ffac, name] = (
                np.fft.rfft(polarisation * grid.t_window_fine)[: base.n_omega]
                * grid.dt_fine
            )
            if name == "band":
                diagnostics = passes.plasma_integration_diagnostics(
                    field[None, None, :], (coefficients,), (0.0,), grid.dt_fine, rule
                )
    reference = (4 * spectra[96.0, "trapezoid"] - spectra[48.0, "trapezoid"]) / 3
    check = (4 * spectra[48.0, "trapezoid"] - spectra[24.0, "trapezoid"]) / 3
    ratio = omega / base.omega_max
    active = base.active_mask
    total = float(np.linalg.norm(reference[active]) ** 2)

    def row(axis, mask):
        norm = float(np.linalg.norm(reference[mask]))

        def error(spectrum):
            return float(np.linalg.norm(spectrum[mask] - reference[mask])) / norm

        return {
            "series": case,
            "axis": axis,
            "band_rule": error(spectra[6.0, "band"]),
            "trapezoid": error(spectra[6.0, "trapezoid"]),
            "reference": error(check),
            "share": norm**2 / total,
        }

    rows = [
        row(
            f"{low:.2f}-{high:.2f}",
            active & (ratio >= low) & ((ratio < high) | (high == BANDS[-1][1])),
        )
        for low, high in BANDS
    ]
    rows.append(row("active", active))
    rows.append(
        {
            "series": case,
            "axis": "work identity",
            "band_rule": work["band"],
            "trapezoid": work["trapezoid"],
            "reference": None,
            "share": None,
        }
    )
    assert diagnostics is not None
    undershoot, work_defect, _ = passes.plasma_integration_statistics(diagnostics)
    return rows, undershoot, work_defect


@pytest.mark.parametrize("case", list(CASES))
def test_the_band_rule_meets_its_targets_on_ionising_fields(request, case):
    rows, undershoot, work_defect = study(case)
    for row in rows:
        if row["axis"] == "work identity":
            assert row["band_rule"] <= WORK_TARGET, row
        elif row["axis"] == "active":
            assert row["band_rule"] <= ACTIVE_TARGET, row
        elif row["share"] >= FLOOR:
            assert row["band_rule"] <= BAND_TARGET, row
        # The reference resolves every asserted band an order better than the target.
        if row["axis"] == "active" or (row["share"] or 0.0) >= FLOOR:
            assert row["reference"] <= BAND_TARGET / 10, row
    # Section 4.5: the work defect and the undershoot on resolved fields.
    assert work_defect <= WORK_DEFECT_TARGET
    assert undershoot <= 1e-3
    record(PLASMA_BAND_RESPONSE, request=request, rows=rows)

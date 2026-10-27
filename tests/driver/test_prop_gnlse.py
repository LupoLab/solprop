# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""`prop_gnlse`: milestone M1, and the keyword surface that reaches it."""

import logging
import os
import subprocess
import sys
from contextlib import nullcontext
from typing import Any, cast

import h5py
import numpy as np
import pytest

from bounds import departure
from march_bounds import FIXED_ORDER, assert_matched, simulation_bound
from selected_backend import selected_backend
from solprop import fourier
from solprop.backends.registry import select
from solprop.constants import C
from solprop.driver import Simulation, prop_gnlse
from solprop.driver.simulation import _dispersion_table, _gnlse_mode, _shot_noise
from solprop.exceptions import DriverError, ModeError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import IndexTable, MarcatiliMode
from solprop.modes.base import beta
from solprop.nonlinear.gnlse import (
    GnlseProblem,
    SpecifiedMode,
    _ModeBeta,
    attenuation_from_db,
)
from solprop.output import HDF5Output, MemoryOutput
from solprop.output.schema import serialise
from solprop.physdata import Mixture
from solprop.profiles import Linear
from solprop.stepper import FixedSchedule


def item(parent, *names) -> Any:
    """Return one h5py object, loosely typed so that a test may inspect it."""
    value = parent
    for name in names:
        value = value[name]
    return value


def recorded(handler) -> Any:
    """Return the metadata a handler holds, which it only lacks before it is opened."""
    assert handler.metadata is not None
    return handler.metadata


# A cheap run for the keyword tests: a millimetre of fibre on the 1024-bin grid, where
# the pulse barely evolves and the surface is what is under test.
SHORT = 1e-3


def cheap_settings(**keywords):
    """Return the keywords of a millimetre of the soliton fibre."""
    return {
        "gamma": 0.11,
        "length": SHORT,
        "betas": [0.0, 0.0, -11.83e-27],
        "lambda0": 835e-9,
        "wavelength_limits": (500e-9, 2e-6),
        "trange": 1e-12,
        "width": 100e-15,
        "peak_power": 9.8,
        "pulseshape": "sech",
        "shock": False,
        "rtol": 1e-8,
        "atol": 1e-18,
        "n_save": 2,
        "backend": selected_backend(),
    } | keywords


def cheap(**keywords):
    """Run a millimetre of the soliton fibre through the whole interface."""
    settings = cheap_settings(**keywords)
    return prop_gnlse(
        settings.pop("gamma"),
        settings.pop("length"),
        settings.pop("betas"),
        **settings,
    )


def prescribed(save_planes=(SHORT,)):
    """Return the keywords of a prescribed ten-step fixed-order march of :func:`cheap`.

    Two runs of it are one discrete march, compared within its rounding budget
    (SEP 25): no step or order is left for rounding to choose.
    """
    return {
        "schedule": FixedSchedule.uniform(
            0.0, SHORT, SHORT / 20, save_planes=list(save_planes)
        ),
        "integrator": FIXED_ORDER,
    }


def cheap_bound(**keywords):
    """Return the matched-march budget of :func:`cheap` with `keywords`."""
    settings = cheap_settings(**keywords)
    for name in ("output", "progress"):
        settings.pop(name, None)
    simulation = Simulation.from_gnlse(
        settings.pop("gamma"),
        settings.pop("length"),
        settings.pop("betas"),
        **settings,
    )
    return simulation_bound(simulation)


# ------------------------------------------------------------------- milestone M1
#
# What these two test is the interface: that the keywords a user types assemble the
# problem the backend is then handed, and that the driver's loop hands back exactly
# what the backend produced. That is settled to rounding at a few hundred steps -- a
# converged recurrence is not needed to test a keyword surface -- and at converged
# step counts the two runs below would cost two minutes between them. The recurrence
# itself, at the step counts that converge it, is milestone M1's definition and is
# measured against the backend directly in
# ``tests/backends/test_numpy_ref_soliton.py``.

# A few hundred steps: enough that the pulse really propagates, cheap enough for a tier
# that runs in the gate.
MEDIUM_STEPS = 400


def interface_and_backend(simulation):
    """Return the handler, its exit plane and the backend's, all from one translation.

    Both sides step the very same :class:`~solprop.translate.problem.KernelProblem` on
    its prescribed schedule at a fixed order, and ``fields("fft")`` undoes the stored
    optics convention with a conjugation and a per-bin sign, neither of which loses a
    bit (``docs/dev/testing.md`` section 3). The two runs are one march, so they agree
    within its rounding budget (SEP 25), with the soliton's nonlinear phase ``Phi``
    taken at the launch as the budget's propagation growth.
    """
    problem = simulation.translate()
    handler = simulation.run()
    backend = select(
        simulation.settings.backend,
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        direct = backend.observe().require_field()[0].copy()
    finally:
        backend.close()
    return handler, handler.fields("fft")[-1], direct


@pytest.mark.medium
def test_the_third_order_soliton_runs_the_same_through_the_interface(
    soliton, test_backend_name
):
    """M1's N = 3 case from the keywords a user types, and its recurrence with it.

    The stored spectrum is in the optics convention, so the global phase after one
    period is ``+pi/4``, the conjugate of the ``-pi/4`` the backend works in (ADR-017).
    At 400 fixed steps the recurrence is the scheme's own truncation error rather than
    the ten digits the slow test reaches, which is enough to say the whole surface --
    grid, pulse, mapping, schedule, output -- reached the solver intact.
    """
    simulation = Simulation.from_gnlse(
        soliton.gamma,
        soliton.period,
        [0.0, 0.0, soliton.beta2],
        lambda0=soliton.wavelength,
        wavelength_limits=(500e-9, 2e-6),
        trange=4e-12,
        width=soliton.tau0,
        peak_power=soliton.peak_power(3),
        pulseshape="sech",
        shock=False,
        schedule=FixedSchedule.uniform(
            0.0, soliton.period, soliton.period / MEDIUM_STEPS
        ),
        integrator=FIXED_ORDER,
        dz_max=soliton.period,
        dz_init=soliton.period / MEDIUM_STEPS,
        n_save=2,
        backend=test_backend_name,
        noise=None,
    )
    handler, stored, direct = interface_and_backend(simulation)
    assert_matched(stored, direct, simulation_bound(simulation))
    start, final = handler["field"][0, 0, 0], handler["field"][-1, 0, 0]
    # Measured 1.0862e-3 over 400 steps, and the phase to 4.51e-4. The backend's own
    # ladder reads 1.09e-3 at the same step count on the same grid
    # (tests/backends/test_numpy_ref_soliton.py), so the interface reaches the solver
    # the ladder measures and adds nothing; the bounds leave a factor of three and four.
    assert soliton.error(final, start) < 3e-3
    assert soliton.phase(final, start) == pytest.approx(
        soliton.recurrence_phase, abs=2e-3
    )


@pytest.mark.medium
def test_the_sixth_order_soliton_runs_the_same_through_the_interface(
    soliton, wide_grid, test_backend_name
):
    """The same, on the band M1's ladder is measured on and with a grid passed in.

    A tenth of the period rather than a whole one: at 400 steps a full N = 6 period is
    nowhere near converged, and an unconverged run says nothing about an interface that
    a converged tenth does not say better. The ladder that *is* M1 runs 8000 and 16000
    steps over the whole period, on the backend, in the slow lane.

    Energy is asserted beside the interface's agreement because it is the one physical
    quantity a short run of this case still pins: the Kerr term conserves it exactly and
    the only sink is the spectral collar, which a band from 350 nm to 3 um keeps the
    compressing pulse away from.
    """
    stop = soliton.period / 10
    simulation = Simulation.from_gnlse(
        soliton.gamma,
        stop,
        [0.0, 0.0, soliton.beta2],
        lambda0=soliton.wavelength,
        grid=wide_grid,
        width=soliton.tau0,
        peak_power=soliton.peak_power(6),
        pulseshape="sech",
        shock=False,
        schedule=FixedSchedule.uniform(0.0, stop, stop / MEDIUM_STEPS),
        integrator=FIXED_ORDER,
        dz_max=stop,
        dz_init=stop / MEDIUM_STEPS,
        n_save=2,
        backend=test_backend_name,
        noise=None,
    )
    handler, stored, direct = interface_and_backend(simulation)
    assert_matched(stored, direct, simulation_bound(simulation))
    energy = handler["stats/energy"][:, 0, 0]
    # Measured 1.04e-6 relative over the run, and the bound leaves an order of
    # magnitude. Energy is the robust quantity at this step count: the state itself is
    # a thousandth from converged, and its norm a millionth.
    assert energy[-1] / energy[0] == pytest.approx(1.0, rel=1e-5)
    # The run is a real one: an N = 6 pulse over a tenth of a period broadens its
    # spectrum by far more than any tolerance, so a driver that stored the entrance
    # twice, or a backend that never stepped, fails here.
    entrance = handler.fields("fft")[0]
    # Measured 1.70: the exit spectrum is further from the entrance than the entrance
    # is from zero.
    assert float(np.linalg.norm(stored - entrance) / np.linalg.norm(entrance)) > 1.0


# --------------------------------------------------------------- the fibre keywords


def test_a_tabulated_beta_matches_the_taylor_series_it_was_sampled_from(soliton):
    grid = EnvelopeGrid.from_limits(SHORT, soliton.wavelength, (500e-9, 2e-6), 1e-12)
    series = SpecifiedMode.from_taylor([0.0, 0.0, soliton.beta2], grid.omega0)
    nodes = np.unique(grid.omega[grid.active_mask])
    taylor = cheap(grid=grid, wavelength_limits=None, trange=None)
    table = cheap(
        grid=grid,
        wavelength_limits=None,
        trange=None,
        betas=None,
        beta=(nodes, series.beta(nodes)),
    )
    difference = np.linalg.norm(
        table["field"][-1] - taylor["field"][-1]
    ) / np.linalg.norm(taylor["field"][-1])
    # A not-a-knot cubic spline through the samples of a quadratic reproduces it, so
    # what is left is the spline solve's round-off on numbers of size beta ~ 1e7 over a
    # millimetre of propagation. Measured 2e-14.
    assert difference < 1e-12


def test_a_callable_beta_is_sampled_onto_the_grid(soliton):
    grid = EnvelopeGrid.from_limits(SHORT, soliton.wavelength, (500e-9, 2e-6), 1e-12)
    series = SpecifiedMode.from_taylor([0.0, 0.0, soliton.beta2], grid.omega0)
    taylor = cheap(grid=grid, wavelength_limits=None, trange=None)
    called = cheap(
        grid=grid,
        wavelength_limits=None,
        trange=None,
        betas=None,
        beta=series.beta,
    )
    assert np.allclose(called["field"][-1], taylor["field"][-1], rtol=1e-12, atol=0.0)


def test_the_loss_matches_the_decibel_conversion():
    """A lossy, linear fibre: the energy falls by exactly exp(-alpha z)."""
    loss = 30.0  # dB/m
    handler = cheap(gamma=0.0, loss_db_per_m=loss)
    energy = handler["stats/energy"][:, 0, 0]
    expected = np.exp(-attenuation_from_db(loss) * SHORT)
    # The nonlinearity conserves energy exactly, so the ratio is the linear operator's
    # own: the tolerance is the Chebyshev table's target accuracy of the operator,
    # 1e-12, not the stepper's.
    assert energy[-1] / energy[0] == pytest.approx(expected, rel=1e-10)


def test_the_polarisation_taper_can_be_switched_on():
    """The polarisation taper reaches the interface, and it changes the answer.

    Off by default: the spectral collar absorbs the field through the linear operator
    either way, and tapering the polarisation as well costs a systematic error of the
    size of the field's content in the collar. Here the pulse is narrow and the run is
    short, so the difference is small but not zero.
    """
    without = cheap()
    with_taper = cheap(polarisation_taper=True)
    difference = np.linalg.norm(
        with_taper["field"][-1] - without["field"][-1]
    ) / np.linalg.norm(without["field"][-1])
    assert difference > 0.0


def test_the_shock_term_can_be_switched_on():
    """`shock=True` is the prefactor's omega dependence, and it changes the answer."""
    without = cheap(shock=False)
    with_shock = cheap(shock=True)
    assert not np.allclose(with_shock["field"][-1], without["field"][-1])


# --------------------------------------------------------------- the pulse keywords


@pytest.mark.parametrize(
    ("pulseshape", "duration"),
    [("sech", {"width": 100e-15}), ("gauss", {"fwhm": 176e-15})],
)
def test_both_pulse_shapes_run(pulseshape, duration):
    handler = cheap(pulseshape=pulseshape, **({"width": None} | duration))
    assert handler.n_saved == 2
    # peak_power carries one value per mode, so the stored array is (n_z, n_modes).
    assert float(handler["stats/peak_power"][0, 0]) == pytest.approx(9.8, rel=1e-3)


def test_the_energy_and_the_peak_power_specify_the_same_sech_pulse():
    """For a sech**2 of natural width tau0, the energy is 2 P0 tau0."""
    peak = 9.8
    width = 100e-15
    by_power = cheap(peak_power=peak, width=width)
    by_energy = cheap(peak_power=None, energy=2.0 * peak * width, width=width)
    # `2 P0 tau0` is the continuum energy while the pulse is scaled on the samples, so
    # the two requests differ by the sech's truncated tails on a 1 ps window: the
    # sampled energy of the peak-scaled pulse is 8.0e-8 below the analytic value, which
    # is the difference measured here (7.9e-8 relative in amplitude).
    assert np.allclose(by_energy["field"][-1], by_power["field"][-1], rtol=1e-7)
    assert not np.allclose(by_energy["field"][-1], by_power["field"][-1], rtol=1e-9)


def test_a_group_delay_phase_moves_the_pulse_later(soliton):
    """`phases` is the optics-convention Taylor phase, so a positive delay is later."""
    delay = 120e-15
    grid = EnvelopeGrid.from_limits(SHORT, soliton.wavelength, (500e-9, 2e-6), 1e-12)
    plain = cheap(grid=grid, wavelength_limits=None, trange=None)
    delayed = cheap(grid=grid, wavelength_limits=None, trange=None, phases=[0.0, delay])
    peaks = [
        grid.t[
            int(
                np.argmax(
                    np.abs(
                        fourier.inverse(handler.fields("fft")[0, 0, 0], kind="envelope")
                    )
                )
            )
        ]
        for handler in (plain, delayed)
    ]
    # The shift is realised on the grid, so it lands on the nearest sample: the
    # tolerance is the coarse spacing dt, 0.98 fs here.
    assert peaks[1] - peaks[0] == pytest.approx(delay, abs=grid.dt)


# --------------------------------------------------------------- the run keywords


def test_the_noise_is_reproducible_from_its_seed():
    # The same seed draws the same realisation; on a prescribed fixed-order march its
    # two runs agree within the march's rounding budget, and another seed's far
    # beyond it (SEP 25 4.5).
    once = cheap(noise="input", seed=5, **prescribed())
    again = cheap(noise="input", seed=5, **prescribed())
    other = cheap(noise="input", seed=6, **prescribed())
    bound = cheap_bound(noise="input", seed=5, **prescribed())
    assert_matched(again["field"][-1], once["field"][-1], bound)
    assert departure(other["field"][-1], once["field"][-1]) > 100 * bound
    record = recorded(once).noise
    assert record.seed == 5
    assert record.model == "input"


def test_the_modified_noise_model_carries_a_phase_at_every_plane(tmp_path):
    path = tmp_path / "noisy.h5"
    handler = cheap(noise="modified", seed=5, output=path)
    assert recorded(handler).noise.model == "modified"
    # The noise phase travels beside the field so that processing can restore the
    # background the propagating state never carried (ADR-008). The handler is closed
    # when prop_gnlse returns, so the file is read directly here; an open handler
    # answers the 'noise_phase' key.
    with h5py.File(path, "r") as file:
        assert item(file, "noise_phase").shape == item(file, "field").shape
        assert item(file, "noise_phase").attrs["spectral_convention"] == "fft"


def test_n_save_lays_out_the_same_planes_as_save_planes_would():
    planes = np.linspace(0.0, SHORT, 5)[1:]
    counted = cheap(n_save=5, **prescribed(planes))
    listed = cheap(n_save=None, save_planes=planes, **prescribed(planes))
    # The positions are the layout: exact. The fields are one march run twice.
    assert np.array_equal(counted["z"], listed["z"])
    assert_matched(
        listed["field"], counted["field"], cheap_bound(n_save=5, **prescribed(planes))
    )


def test_an_output_path_writes_a_file_in_the_optics_convention(tmp_path):
    path = tmp_path / "run.h5"
    handler = cheap(output=path)
    # The handler is an HDF5Output, which keeps the path it was built for; the
    # annotation is the Output protocol, which does not promise one.
    assert isinstance(handler, HDF5Output)
    assert handler.path == path
    with h5py.File(path, "r") as file:
        assert file.attrs["spectral_convention"] == "optics"
        assert item(file, "field").attrs["spectral_convention"] == "optics"
        assert item(file, "field").shape[0] == 2
        assert "gnlse" in str(item(file, "meta", "json")[()])


def test_a_handler_can_be_given_directly():
    handler = MemoryOutput()
    assert cheap(output=handler) is handler
    assert handler.n_saved == 2


# ------------------------------------------------------------------ the refusals


def recorded_fibre(handler):
    """Return the GNLSE mapping the run recorded in its metadata."""
    metadata = handler.metadata
    assert metadata is not None
    return cast(GnlseProblem, metadata.configuration["gnlse"])


def test_raman_runs_and_its_fraction_reaches_the_response():
    """``raman=True`` propagates, and the share it takes is the one asked for."""
    handler = cheap(raman=True, raman_fraction=0.25)
    fibre = recorded_fibre(handler)
    raman = fibre.raman
    assert raman is not None
    assert fibre.raman_fraction == 0.25
    assert raman.fraction == 0.25
    assert handler.n_saved == 2
    # The record a run is resumed and compared from carries the response too.
    recorded = cast(dict[str, Any], serialise(fibre))["raman"]
    assert recorded["fraction"] == 0.25


def test_raman_takes_its_share_out_of_the_kerr_term():
    """The Kerr coefficient is reduced by exactly the fraction the Raman term takes,
    and with the fraction left out only when the Raman term is on."""
    with_raman = recorded_fibre(cheap(raman=True, raman_fraction=0.25))
    without = recorded_fibre(cheap(raman=False))
    assert with_raman.chi3 == pytest.approx(0.75 * without.chi3, rel=1e-14)
    assert without.raman is None


def test_a_given_fraction_with_raman_off_reaches_the_kerr_term():
    """An explicit fraction with Raman off is held back, and no response carries it."""
    held = recorded_fibre(cheap(raman=False, raman_fraction=0.25))
    with_raman = recorded_fibre(cheap(raman=True, raman_fraction=0.25))
    assert held.raman is None
    assert held.raman_fraction == 0.25
    assert held.chi3 == with_raman.chi3


def test_a_field_grid_is_refused_by_name(soliton):
    grid = FieldGrid.from_limits(SHORT, soliton.wavelength, (500e-9, 2e-6), 1e-12)
    with pytest.raises(DriverError, match="envelope grid"):
        cheap(grid=grid, wavelength_limits=None, trange=None)


def test_a_gaussian_asked_for_by_its_natural_width_is_refused():
    with pytest.raises(DriverError, match="intensity fwhm"):
        cheap(pulseshape="gauss", width=100e-15)


def test_an_unknown_pulse_shape_is_refused():
    with pytest.raises(DriverError, match="'gauss' or 'sech'"):
        cheap(pulseshape="lorentz", width=None, fwhm=100e-15)


def test_an_unknown_noise_model_is_refused(narrow_grid):
    # Through the helper directly: the interface annotates the model as a Literal, so
    # the test session's hook rejects a bad string before the driver's own check is
    # reached, and that check has to hold in production too.
    with pytest.raises(DriverError, match="None, 'input' or 'modified'"):
        _shot_noise("thermal", narrow_grid, 3)


@pytest.mark.parametrize(
    "keywords",
    [{"betas": None}, {"beta": (np.zeros(4), np.zeros(4))}],
    ids=["neither", "both"],
)
def test_exactly_one_dispersion_must_be_given(keywords):
    with pytest.raises(DriverError, match="exactly one of betas"):
        cheap(**keywords)


def test_a_dispersion_that_is_neither_a_table_nor_a_callable_is_refused(narrow_grid):
    # Through the helper directly, for the reason above.
    with pytest.raises(DriverError, match="pair of arrays or a callable"):
        _dispersion_table("beta", narrow_grid)


def test_a_grid_cannot_be_left_unspecified():
    with pytest.raises(DriverError, match="wavelength_limits and trange"):
        cheap(wavelength_limits=None)


def test_both_ways_of_asking_for_planes_at_once_are_refused():
    with pytest.raises(DriverError, match="not both"):
        cheap(n_save=4, save_planes=[SHORT])


def test_a_run_must_store_more_than_the_entrance():
    with pytest.raises(DriverError, match="at least 2"):
        cheap(n_save=1)


# ------------------------------------------------------------- the periodic progress


def test_a_short_period_puts_a_progress_line_in_the_log(caplog):
    """The period is wall time, so a run this short needs one near zero to emit."""
    with caplog.at_level(logging.INFO, logger="solprop.log"):
        cheap(progress_period=1e-9)
    lines = [record.getMessage() for record in caplog.records]
    assert sum(line.startswith("progress ") for line in lines) >= 1
    assert sum(line.startswith("finished at z = ") for line in lines) == 1


def test_a_period_of_zero_leaves_only_the_plane_and_closing_lines(caplog):
    with caplog.at_level(logging.INFO, logger="solprop.log"):
        cheap(progress_period=0.0)
    lines = [record.getMessage() for record in caplog.records]
    assert not any(line.startswith("progress ") for line in lines)
    assert sum(line.startswith("finished at z = ") for line in lines) == 1


def test_the_default_period_is_long_enough_that_a_short_run_says_nothing(caplog):
    """A millisecond of fibre is not a run anyone needs told about."""
    with caplog.at_level(logging.INFO, logger="solprop.log"):
        cheap()
    assert not any(
        record.getMessage().startswith("progress ") for record in caplog.records
    )


# ------------------------------------------------------------------------ the bar


def test_under_capture_the_default_logs_lines_and_draws_no_bar(caplog, capsys):
    """pytest's captured standard error is not a terminal, so ``auto`` means lines."""
    with caplog.at_level(logging.INFO, logger="solprop.log"):
        cheap(progress_period=1e-9)
    assert any(r.getMessage().startswith("progress ") for r in caplog.records)
    assert "%|" not in capsys.readouterr().err


def test_progress_bar_draws_on_standard_error_when_asked(capsys):
    cheap(progress="bar")
    assert "100.0%|" in capsys.readouterr().err


def test_progress_none_reports_nothing(caplog, capsys):
    with caplog.at_level(logging.INFO, logger="solprop.log"):
        cheap(progress="none")
    assert not any(
        r.getMessage().startswith(("progress ", "finished at ")) for r in caplog.records
    )
    assert "%|" not in capsys.readouterr().err


# ----------------------------------------------------------- tabulated fibre (37)


def solid_core_table(*, with_area=True):
    omega0 = 2 * np.pi * C / 835e-9
    area = 1e-10
    coefficients = [1.45 * omega0 / C, 1.45 / C, -11.83e-27]
    mode = SpecifiedMode.from_taylor(
        coefficients, omega0, loss=attenuation_from_db(0.2), aeff=area
    )
    omega = np.linspace(0.3e15, 4.5e15, 200)
    table = IndexTable(
        omega, mode.neff(omega), area=np.full(200, area) if with_area else None
    )
    return table, mode, coefficients


def test_index_table_fibre_matches_taylor_mapping():
    table, mode, coefficients = solid_core_table()
    omega0 = 2 * np.pi * C / 835e-9
    length = 0.01
    schedule = FixedSchedule.uniform(0.0, length, length / 8)
    settings = {
        "length": length,
        "betas": coefficients,
        "loss_db_per_m": 0.2,
        "schedule": schedule,
    }
    reference = cheap(**settings)
    actual = cheap(
        **(
            settings
            | {
                "gamma": None,
                "betas": None,
                "loss_db_per_m": None,
                "fibre": table,
                "n2": 0.11 * C * mode.aeff() / omega0,
            }
        )
    )
    grid = EnvelopeGrid.from_limits(length, 835e-9, (500e-9, 2e-6), 1e-12)
    omega = grid.omega[grid.active_mask]
    table_mode = SpecifiedMode.from_index_table(table, aeff=mode.aeff())
    # Error in the co-moving linear operator bounds accumulated phase/loss;
    # exp(3 gamma P L) allows the cubic nonlinearity's Lipschitz amplification.
    # The remaining term counts the eight steps' FFT reductions over the
    # transform length grid.n_t, at 128 eps per sample as the channel ladder
    # allocates for a radial node.
    phase = table.beta(omega) - mode.beta(omega)
    phase -= table.beta([omega0])[0] - mode.beta(np.array([omega0]))[0]
    phase -= (
        table.dbeta_domega([omega0])[0]
        - mode.dispersion_derivative(np.array([omega0]))[0]
    ) * (omega - omega0)
    operator_error = np.max(abs(phase)) + 0.5 * np.max(
        abs(table.alpha(omega) - mode.alpha(omega))
    )
    bound = (
        length * operator_error * np.exp(3 * 0.11 * 9.8 * length)
        + 128 * 8 * grid.n_t * np.finfo(float).eps
    )
    error = float(
        np.max(abs(actual.fields("fft") - reference.fields("fft")))
        / np.max(abs(reference.fields("fft")))
    )
    assert 0 < error < bound
    assert np.all(np.isfinite(actual["field"]))
    assert table_mode.aeff() == mode.aeff()
    mapping = recorded(actual).configuration["gnlse"]
    assert isinstance(mapping, GnlseProblem)
    assert mapping.prefactor_index == 1.0
    assert mapping.gamma == pytest.approx(
        0.11, rel=8 * np.finfo(float).eps, abs=0
    )  # Two SI conversions.


def test_supplied_mode_and_table_without_area_use_gamma():
    table, mode, _ = solid_core_table(with_area=False)
    for fibre in (table, mode):
        simulation = Simulation.from_gnlse(
            0.11,
            0.001,
            fibre=fibre,
            lambda0=835e-9,
            wavelength_limits=(500e-9, 2e-6),
            trange=1e-12,
            width=100e-15,
            peak_power=9.8,
            pulseshape="sech",
            n_save=2,
            noise=None,
        )
        mapped = simulation.configuration["gnlse"]
        assert isinstance(mapped, GnlseProblem)
        assert mapped.aeff == (1.0 if fibre is table else mode.aeff())
        if fibre is mode:
            assert mapped.mode is not mode
            assert isinstance(mapped.mode, SpecifiedMode)
            assert isinstance(mapped.mode.dispersion, _ModeBeta)
            assert mapped.mode.dispersion.mode is mode
        output = simulation.run()
        assert np.all(np.isfinite(output["field"]))


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"gamma": 0.11, "n2": 1e-20}, "exactly one of gamma and n2"),
        ({"gamma": None}, "exactly one of gamma and n2"),
        (
            {"gamma": None, "n2": 1e-20, "fibre": None, "betas": [0.0, 0.0, -1e-26]},
            "n2 requires fibre",
        ),
        ({"loss_db_per_m": 0.0}, "fibre carries its loss"),
        ({"loss_db_per_m": 0.2}, "fibre carries its loss"),
        ({"betas": [0.0]}, "exactly one of betas, beta and fibre"),
        ({"beta": np.ones_like}, "exactly one of betas, beta and fibre"),
    ],
)
def test_gnlse_fibre_refusals(overrides, message):
    table, _, _ = solid_core_table()
    options = {"gamma": 0.11, "betas": None, "fibre": table}
    with pytest.raises(DriverError, match=message):
        cheap(**(options | overrides))


def test_n2_without_table_area_is_refused():
    table, _, _ = solid_core_table(with_area=False)
    with pytest.raises(DriverError, match="effective-area column"):
        cheap(gamma=None, betas=None, fibre=table, n2=1e-20)


def test_a_fibre_whose_geometry_moves_carries_both_paths():
    # Admitted by construction: the wrapping samples the source's dispersion and
    # its area together, over the positions the source itself declares.
    mode = MarcatiliMode(Linear(15e-6, 10e-6, 1.0), cladding=1.45)
    fibre = cast(
        "SpecifiedMode", _gnlse_mode(mode, 2 * np.pi * C / 2.0e15, needs_area=False)
    )
    assert not fibre.z_constant
    assert not fibre.dispersion_is_constant
    assert fibre.aeff(0.5) == pytest.approx(mode.aeff(0.5))
    assert np.array_equal(
        fibre.beta(np.array([2.0e15]), 0.5), beta(mode, np.array([2.0e15]), 0.5)
    )


@pytest.mark.parametrize("use_mode", [False, True])
def test_index_table_gnlse_metadata_hash_across_processes(use_mode):
    omega = np.linspace(0.3e15, 4.5e15, 200)
    table = IndexTable(omega, np.full(200, 1.45 + 1e-9j), area=np.full(200, 1e-10))
    fibre = (
        MarcatiliMode(
            15e-6, mixture=Mixture.fill({"Ar": 1.0}, pressure=1.0), cladding=1.45
        )
        if use_mode
        else table
    )
    # The mode route kept the registry's gas-fit diagnostic; optmat's argon is
    # qualified over the band (SEP 3, Phase 4a). The hash is what this test is about.
    context = nullcontext()
    with context:
        expected = (
            Simulation.from_gnlse(
                None,
                0.001,
                fibre=fibre,
                n2=1e-20,
                lambda0=835e-9,
                wavelength_limits=(500e-9, 2e-6),
                trange=1e-12,
                fwhm=100e-15,
                energy=1e-12,
                n_save=2,
                noise=None,
            )
            .metadata()
            .configuration_hash()
        )
    script = """
import numpy as np
from solprop.modes import IndexTable, MarcatiliMode
from solprop.physdata import Mixture
from solprop.driver import Simulation
omega = np.linspace(.3e15, 4.5e15, 200)
table = IndexTable(omega, np.full(200, 1.45 + 1e-9j), area=np.full(200, 1e-10))
simulation = Simulation.from_gnlse(None, .001, fibre=table, n2=1e-20,
    lambda0=835e-9, wavelength_limits=(500e-9, 2e-6), trange=1e-12,
    fwhm=100e-15, energy=1e-12, n_save=2)
print(simulation.metadata().configuration_hash())
"""
    if use_mode:
        script = script.replace(
            "fibre=table",
            "fibre=MarcatiliMode(15e-6, "
            "mixture=Mixture.fill({'Ar': 1.}, pressure=1.), cladding=1.45)",
        )
    # One child compared with this interpreter, the house pattern of
    # test_metadata_hash_is_stable_across_processes: a different hash seed is the
    # direct test that no builtin hash() is involved, since PYTHONHASHSEED salts
    # str.__hash__ per interpreter and would change the answer.
    process = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ) | {"PYTHONHASHSEED": "12345"},
    )
    assert process.stdout.strip() == expected
    assert len(expected) == 64


@pytest.mark.parametrize("area", [0.0, -1.0, np.nan, np.inf])
def test_external_fibre_invalid_area_is_refused(area):
    _, mode, _ = solid_core_table()
    # An external Mode need not implement SpecifiedMode's constructor
    # validation. Corrupt this fixture to exercise the driver's protocol boundary.
    object.__setattr__(mode, "effective_area", area)
    with pytest.raises(DriverError, match="effective area must be positive and finite"):
        cheap(betas=None, fibre=mode)


def test_table_support_includes_active_band_collars():
    table, _, _ = solid_core_table()
    grid = EnvelopeGrid.from_limits(SHORT, 835e-9, (500e-9, 2e-6), 1e-12)
    active = grid.omega[grid.active_mask]
    assert table.omega[0] < active.min() < active.max() < table.omega[-1]
    # User limits alone omit the nonzero collar, which the linear operator samples.
    narrow = IndexTable.from_wavelength(
        np.linspace(500e-9, 2e-6, 200), np.full(200, 1.45 + 1e-9j)
    )
    with pytest.raises(ModeError, match="active-band collars"):
        cheap(betas=None, fibre=narrow)


def test_gnlse_mode_fill_is_only_a_linear_source():
    original = MarcatiliMode(
        15e-6,
        mixture=Mixture.fill({"Ar": 1.0}, pressure=1.0),
        cladding=1.45,
    )
    simulation = Simulation.from_gnlse(
        0.11,
        SHORT,
        fibre=original,
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        pulseshape="sech",
        n_save=2,
        noise=None,
    )
    mode = simulation.modes[0]
    assert isinstance(mode, SpecifiedMode)
    assert isinstance(mode.dispersion, _ModeBeta)
    assert mode.dispersion.mode is original
    assert simulation.settings.density == 1.0
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        assert simulation.translate().responses.labels == ("medium",)
        metadata = simulation.metadata()
        assert metadata.modes[0].kind.endswith(".SpecifiedMode")
        assert "medium" not in metadata.configuration
        assert np.all(np.isfinite(simulation.run()["field"]))

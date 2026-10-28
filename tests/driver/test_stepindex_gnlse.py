# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the silica strand through prop_gnlse and the direct Simulation route."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings, prop_gnlse
from solprop.driver.simulation import default_collar
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid
from solprop.modes import IndexTable, StepIndexMode
from solprop.nonlinear.gnlse import problem_from
from solprop.output.memory import MemoryOutput
from solprop.physdata import fused_silica
from solprop.pulses import SechPulse
from solprop.stepper import Multistep, PIController

# The `stepsimplecmp.jl` example's physical settings, reduced to a 1 ps window
# (1024 samples) and no shot noise. Two centimetres in the medium
# tier and the example's fifteen in the slow lane.
MEDIUM_LENGTH = 0.02
SLOW_LENGTH = 0.15
INTERFACE_LENGTH = 0.015


def strand_simulation(backend, length=SLOW_LENGTH, core=None):
    """Build the reduced strand run through the direct Simulation route."""
    mode = StepIndexMode(1.25e-6, core=core, profile_wavelength=835e-9)
    grid = EnvelopeGrid.from_limits(length, 835e-9, (400e-9, 1400e-9), 1e-12)
    fibre = problem_from(
        [0.0], omega_ref=grid.omega0, n2=2.6e-20, aeff=mode.aeff(), raman=True
    )
    return Simulation(
        grid,
        mode,
        list(fibre.responses),
        SechPulse(wavelength=835e-9, fwhm=50e-15, energy=568e-12),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-5, atol=1e-14),
            save_planes=[length],
            dz_max=0.001,
            dz_init=1.5e-5,
            density=1.0,
            shock=True,
            prefactor_index=1.0,
            backend=backend,
            # The collar prop_gnlse gives the same fibre, so the two routes are one
            # problem; the strand's supercontinuum reaches the window's edges, which
            # its narrow 1 ps window does not mean to hold (SEP 21).
            absorber=default_collar(grid),
            boundary_check=None,
        ),
    )


def spectral_change(fields):
    """Return the relative change of the spectral intensity across the run."""
    entrance, exit_field = np.abs(fields[0]) ** 2, np.abs(fields[-1]) ** 2
    return float(np.linalg.norm(exit_field - entrance) / np.linalg.norm(entrance))


@pytest.mark.medium
def test_stepindex_reduced_strand_propagates_on_the_reference_backend():
    """Two centimetres on `numpy_ref`, the medium tier's share of the strand.

    The duplicate `numba_cpu` copy is the slow case below, as
    ``docs/dev/testing.md`` section 2 prescribes: the parity test stands in for a
    copy already known to agree.
    """
    # The window's collar reaches silica's 2.73 um OH band, where optmat's core is
    # taken as transparent beyond its bound (SEP 3, Q3), and says so once.
    with pytest.warns(MaterialRangeWarning, match="taken as transparent"):
        simulation = strand_simulation("numpy_ref", MEDIUM_LENGTH)
        fields = simulation.run(progress="none").fields("fft")
    assert np.all(np.isfinite(fields))
    # Measured 1.0396 at 2 cm, against 1.0358 at 1.5 cm and the > 0.1 the
    # fifteen-centimetre case asserted: the evolution is genuinely nonlinear,
    # not a linear phase.
    assert spectral_change(fields) > 0.9


@pytest.mark.slow
def test_stepindex_strand_propagates_on_both_cpu_backends(monkeypatch):
    """The example's full fifteen centimetres, both CPU backends."""
    # FFTW_MEASURE times its candidate plans, so under machine load it can pick another
    # plan, whose rounding the adaptive run then carries through every step: the bound
    # below was exceeded once by that alone. ESTIMATE chooses without timing, so the
    # compiled run is the same whatever the load.
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")
    # The window's collar reaches silica's 2.73 um OH band, where optmat's core is
    # taken as transparent beyond its bound (SEP 3, Q3), and each run says so once.
    with pytest.warns(MaterialRangeWarning, match="taken as transparent"):
        first = strand_simulation("numpy_ref").run(progress="none")
    with pytest.warns(MaterialRangeWarning, match="taken as transparent"):
        second = strand_simulation("numba_cpu").run(progress="none")
    reference, compiled = first.fields("fft"), second.fields("fft")
    assert np.all(np.isfinite(reference))
    # The two backends take the same number of steps (6033 under the default), each
    # within 2.1e-4 of the other's, so what parts them is rounding, carried by the
    # Adams history and its controller through fifteen centimetres of strongly
    # nonlinear evolution: measured 6.9e-9 of the largest component with ESTIMATE
    # plans (5.0e-8 with MEASURE, on a quiet machine); the bound is unchanged.
    assert isinstance(first, MemoryOutput)
    assert isinstance(second, MemoryOutput)
    steps = [run.planes()[-1].step_history for run in (first, second)]
    assert steps[0].n_steps == steps[1].n_steps
    np.testing.assert_allclose(steps[1].h, steps[0].h, rtol=1e-3)
    assert_allclose(
        compiled, reference, rtol=0.0, atol=2e-7 * float(np.max(abs(reference)))
    )
    # Nonlinear evolution must change the spectrum, not just apply a linear phase.
    assert spectral_change(reference) > 0.1


def interface_run(fibre, backend="numpy_ref"):
    """Run the same fibre through the public GNLSE entry point."""
    return prop_gnlse(
        None,
        INTERFACE_LENGTH,
        fibre=fibre,
        n2=2.6e-20,
        lambda0=835e-9,
        wavelength_limits=(400e-9, 1400e-9),
        trange=1e-12,
        fwhm=50e-15,
        energy=568e-12,
        pulseshape="sech",
        shock=True,
        raman=True,
        rtol=1e-5,
        atol=1e-14,
        dz_max=0.001,
        dz_init=1.5e-5,
        n_save=2,
        backend=backend,
        progress="none",
        noise=None,
        integrator=Multistep(),
        boundary_check=None,
    ).fields("fft")


@pytest.mark.slow
def test_stepindex_fibre_reaches_prop_gnlse_directly_and_as_a_table():
    """A step-index mode reaches `prop_gnlse` through the `fibre=` keyword.

    The mode goes through `prop_gnlse(fibre=...)` and separately through a table
    the user builds from it, both against the direct route of the tests above.
    """
    # Malitson's silica, Luna's own `:SiO2` and the core these bounds were measured
    # on. optmat's recommended `fit/SiO2` carries the 2.73 um OH band inside the
    # table's collar, whose real-index structure a 6.4 nm knot spacing resolves only
    # to 8.5e-8 there (7.1e-10 elsewhere, measured); the subject here is the
    # interface, not the band, which the tests above keep.
    core = fused_silica(model="malitson_1965/sellmeier")
    mode = StepIndexMode(1.25e-6, core=core, profile_wavelength=835e-9)
    reference = (
        strand_simulation("numpy_ref", INTERFACE_LENGTH, core)
        .run(progress="none")
        .fields("fft")
    )
    grid = EnvelopeGrid.from_limits(INTERFACE_LENGTH, 835e-9, (400e-9, 1400e-9), 1e-12)
    through_mode = interface_run(mode)
    worst = float(np.max(abs(through_mode - reference)) / np.max(abs(reference)))
    # The same problem assembled twice, so the two agree to rounding: measured
    # 6.43e-12 on macOS arm64 and 3.11e-10 on Linux x86-64 (GCC, FFTW 3.3.11). The
    # bound is 512 eps per sample over the transform length and the eight-stage
    # reductions (9.3e-10), three times the larger.
    assert 0 < worst < 512 * 8 * grid.n_t * np.finfo(float).eps
    # The user's own table of the same fibre, 400 knots over the active band and
    # its collars, with the area column. A lossless eigenproblem has Im n = 0,
    # so the logarithmic loss interpolation is refused and "linear" is required.
    wavelengths = np.linspace(360e-9, 2900e-9, 400)
    table = IndexTable.from_wavelength(
        wavelengths,
        mode.neff(2 * np.pi * C / wavelengths),
        area=np.full(400, mode.aeff()),
        loss_interpolation="linear",
    )
    active = np.sort(grid.omega[grid.active_mask])
    index_error = float(np.max(np.abs(table.neff(active) - mode.neff(active))))
    # Measured: 7.05e-10 in the index over the active band, which is 0.055 1/m
    # of co-moving operator error and 8.3e-4 rad over 1.5 cm, and the exit field
    # then differs by a measured 4.67e-5 of its peak.
    assert index_error < 1e-9
    through_table = interface_run(table)
    difference = float(np.max(abs(through_table - reference)) / np.max(abs(reference)))
    assert 0 < difference < 1e-4

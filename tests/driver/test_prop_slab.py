# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the hollow planar (slab) guide end to end, per unit width, on three backends.

Every run is ten fixed steps of a 10 cm argon-filled gap (2a = 127 um, Akturk et
al. 2009), under a collar weakened far below every bound here so that the explicit
nonlinear-scheme absorber does not cap the step (ADR-025, the pattern of
``test_coherent_capillary.py``). Energies are in **J/m**: the slab is uniform in y.
"""

import contextlib
import dataclasses
import importlib.util
import json
import math
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from bounds import fft_bound, gamma
from march_bounds import FIXED_ORDER, assert_matched, simulation_bound
from modal_provider_cases import GENERAL_Z_END, general_grid, general_guide
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import Simulation, prop_gnlse, prop_waveguide
from solprop.driver.executors import Local
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import (
    AdiabaticTaperWarning,
    DriverError,
    GasFlowWarning,
    ModeError,
    OutputError,
    SolpropWarning,
)
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.modes import (
    MarcatiliMode,
    SlabMode,
    gaussian_coupling,
    transverse_measure,
)
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.output.hdf5 import open_file
from solprop.output.schema import RunMetadata
from solprop.physdata import n2
from solprop.physdata.selection import MaterialSelection
from solprop.plotting import energy as plot_energy
from solprop.plotting import spec_1d, stats, time_1d
from solprop.processing import get_field, transverse_units
from solprop.processing.scans import (
    _provenance,
    energy_in_band,
    field_at_plane,
    peak_power,
)
from solprop.profiles import Linear
from solprop.stepper import DOPRI5
from solprop.stepper.schedule import FixedSchedule
from solprop.tools import pulse_peak_power

pytestmark = pytest.mark.filterwarnings(
    # The FFT padding reaches beyond the argon fit; the notice is not the subject.
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

LENGTH = 0.1
"""Guide length, m: B is a third of a radian at 0.5 J/m and dispersion negligible."""

STEPS = 10
"""The fixed schedule's steps: the ten-step rule."""

HALF_GAP = 63.5e-6
"""Akturk et al.'s 2a = 127 um, m."""

GRID = EnvelopeGrid.from_limits(LENGTH, 800e-9, (650e-9, 1000e-9), 1.2e-12)
"""One envelope grid for every run, so that launches and bounds share it."""

COLLAR = absorber(GRID, boundary_length=1.0e3)
"""A collar a million times weaker than the default, present but inert (ADR-025)."""

BACKENDS = ["numpy_ref", "numba_cpu"]
if importlib.util.find_spec("solprop_native") is not None:
    BACKENDS.append("native")


def options(**changes):
    return {
        "lambda0": 800e-9,
        "grid": GRID,
        "absorber": COLLAR,
        "fwhm": 100e-15,
        "energy": 0.5,
        "envelope": True,
        "plasma": False,
        "raman": False,
        "n_save": 2,
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        **changes,
    }


def launched(simulation, shares):
    """Replace a single-mode launch by the Gaussian's projection onto every mode."""
    pulse = simulation._field0[0]
    initial = np.stack([math.sqrt(share) * pulse for share in shares])
    launched = simulation.with_input(initial)
    # The record describes the replacement spectrum, not the single-mode pulse.
    assert "pulse" not in launched.configuration
    return launched


def parity_bound(n_nodes):
    """Relative bound on a ten-step final field between two backends.

    The legacy capillary bound (``test_coherent_capillary.legacy_parity_bound``):
    each of the ten steps' seven DOPRI5 stages re-forms the nonlinear term from a
    field the backends built in different orders -- one FFT round trip each way
    and the modal synthesis and projection over at most the rule's nodes -- and
    the errors of the stages add, since the linear step is unit-modulus and the
    nonlinear phase stays below one radian. Written before the runs were compared.
    """
    return STEPS * DOPRI5.n_stages * (2 * fft_bound(GRID.n_to) + 2 * gamma(n_nodes))


# ------------------------------------------------------------------ the physics


@pytest.mark.medium
def test_a_te_slab_run_accumulates_the_closed_form_b_integral():
    """SPM at the pulse peak: B = n2 (omega/c) P' L_eff / (4a/3), per unit width.

    One TE1 mode, 0.5 J/m in a 100 fs Gaussian, argon at 1 bar. The field
    attenuation alpha enters as L_eff = (1 - exp(-2 alpha L)) / 2 alpha. The
    agreement is limited by what the closed form leaves out: the index factors
    n0/n_eff of the true-power amplitude and the prefactor (argon's n - 1 ~ 3e-4),
    and dispersion over L/L_D ~ 1e-3; the ten DOPRI5 steps add far less. Measured
    1.2e-4; the bound is 2e-3.
    """
    mode = SlabMode(HALF_GAP, cladding=1.45)
    output = prop_waveguide(
        [mode],
        LENGTH,
        "Ar",
        1.0,
        **(options()),
        progress="none",
    )
    t, envelope = get_field(output, kind="envelope")
    peak = int(np.argmin(np.abs(t)))
    phase = float(np.angle(envelope[-1, peak]) - np.angle(envelope[0, peak]))
    omega = np.asarray([2 * np.pi * C / 800e-9])
    alpha = float(np.imag(mode.neff(omega))[0] * omega[0] / C)
    effective = (1 - math.exp(-2 * alpha * LENGTH)) / (2 * alpha)
    power = pulse_peak_power(0.5, 100e-15, "gauss")
    # The run's Kerr coefficient is optmat's at the carrier since SEP 3's Phase 4a,
    # 23 % above the registry's: the closed form takes the same one (B 0.358 -> 0.441).
    closed = (
        n2("Ar", pressure=1.0, wavelength=800e-9, selection=MaterialSelection())
        * omega[0]
        / C
        * power
        / mode.aeff()
        * effective
    )
    assert 0.4 < closed < 0.5
    assert_allclose(phase, closed, rtol=2e-3)
    # The launched power per unit width is the pulse's, and the loss is the mode's.
    assert_allclose(float(np.max(np.abs(envelope[0]) ** 2)), power, rtol=1e-12)
    energy = output["stats/energy"].ravel()
    assert_allclose(energy[-1] / energy[0], math.exp(-2 * alpha * LENGTH), rtol=1e-4)


@pytest.mark.medium
def test_the_mode_averaged_slab_is_the_one_mode_run():
    """A_eff = 4a/3 is the overlap of the profile with itself, so the two agree.

    The same launch through the mode-averaged transform (no rule) and through a
    one-mode multimode transform on the Kronrod rule: the Kerr coefficients differ
    only by the rule's rounding, a few eps; measured 3e-16 of the field, bound 1e-13.
    """
    simulation = Simulation.from_waveguide(
        [SlabMode(HALF_GAP, cladding=1.45)],
        LENGTH,
        "Ar",
        1.0,
        **(options()),
    )
    single = simulation.run(progress="none")
    averaged = Simulation(
        simulation.grid,
        simulation.modes,
        simulation.responses,
        simulation._field0,
        settings=dataclasses.replace(
            simulation.settings, quadrature=None, components=None
        ),
        profiles=simulation.profiles,
        configuration=simulation.configuration,
    ).run(progress="none")
    assert averaged.metadata is not None and single.metadata is not None
    assert averaged.metadata.simulation_kind == "mode_averaged"
    assert averaged.metadata.transverse_measure == "per_unit_width"
    scale = float(np.max(np.abs(single["field"])))
    assert_allclose(averaged["field"], single["field"], rtol=0, atol=1e-13 * scale)


@pytest.mark.medium
@pytest.mark.parametrize("taper", [False, True], ids=["constant", "tapered"])
def test_a_projected_multimode_slab_agrees_on_every_backend(taper):
    """TE1/TE3/TE5 (TE1/TE3 tapered), launched as Arnold's 0.735a Gaussian.

    A centred beam reaches the odd orders only. Each backend's final field is
    within :func:`parity_bound` of ``numpy_ref``'s (measured below 1e-15 of a
    1.6e-10 bound), and the energy launched is the coupled share of 0.5 J/m. The
    inert collar cannot stop the modes' 40 fs group-delay walk-off from wrapping,
    which the driver says; ten steps of a 100 fs pulse in a 1.2 ps window never
    reach the edge.
    """
    gap = Linear(HALF_GAP, 50e-6, LENGTH) if taper else HALF_GAP
    orders = (1, 3) if taper else (1, 3, 5)
    modes = [SlabMode(gap, cladding=1.45, order=order) for order in orders]
    shares = gaussian_coupling(modes, 0.735 * HALF_GAP)
    fields = {}
    n_nodes = 0
    for backend in BACKENDS:
        with pytest.warns(SolpropWarning, match="walking off"):
            simulation = Simulation.from_waveguide(
                modes,
                LENGTH,
                "Ar",
                1.0,
                **(options(backend=backend)),
            )
        rule = simulation.settings.quadrature
        assert rule is not None and rule.kind == "slab" and rule.kronrod
        n_nodes = rule.n_nodes
        # A taper couples TE1 to TE3 through <psi_m|d_z psi_n>, which this route
        # omits and says so: each channel follows its local mode adiabatically.
        with (
            pytest.warns(AdiabaticTaperWarning, match="omits the connection")
            if taper
            else contextlib.nullcontext()
        ):
            output = launched(simulation, shares).run(progress="none")
        fields[backend] = output["field"][-1]
        energy = output["stats/energy"]
        assert_allclose(float(np.sum(energy[0])), 0.5 * shares.sum(), rtol=1e-12)
    reference = fields["numpy_ref"]
    scale = float(np.max(np.abs(reference)))
    for backend in BACKENDS[1:]:
        assert float(np.max(np.abs(fields[backend] - reference))) <= (
            parity_bound(n_nodes) * scale
        )


@pytest.mark.medium
def test_a_linear_taper_conserves_power_per_unit_width():
    """Lossless and all but linear, a scale-invariant taper moves no energy.

    The matrices rescale by sqrt(a(0)/a(z)) and the amplitudes are true power
    per metre of y; the energy moves by the integration error alone. The default
    integrator's steps are the controller's: measured 7.1e-14 in 20 steps at
    ``rtol=1e-10`` (2.3e-10 on ten fixed steps, where its history starts at order
    one); bound 1e-12.
    """
    mode = SlabMode(Linear(HALF_GAP, 50e-6, LENGTH), cladding=1.45, loss=False)
    output = prop_waveguide(
        [mode],
        LENGTH,
        "Ar",
        1.0,
        **(options(energy=1e-12, schedule=None, rtol=1e-10)),
        progress="none",
    )
    energy = output["stats/energy"].ravel()
    assert_allclose(energy[-1], energy[0], rtol=1e-12)
    assert output.metadata is not None
    assert output.metadata.modes[0].kind == "solprop.modes.slab.SlabMode"


# ------------------------------------------------------------ records and labels


@pytest.mark.medium
def test_the_run_records_its_measure_and_labels_per_unit_width():
    output = prop_waveguide(
        [SlabMode(HALF_GAP, cladding=1.45)],
        LENGTH,
        "Ar",
        1.0,
        **(options()),
        progress="none",
    )
    metadata = output.metadata
    assert metadata is not None
    assert metadata.transverse_measure == "per_unit_width"
    assert metadata.configuration["transverse_measure"] == "per_unit_width"
    unknown = dataclasses.replace(
        metadata,
        configuration={**metadata.configuration, "transverse_measure": "volume"},
    )
    with pytest.raises(OutputError, match="'area' or 'per_unit_width'"):
        _ = unknown.transverse_measure
    back = RunMetadata.from_json(metadata.to_json())
    assert back.transverse_measure == "per_unit_width"
    assert back.configuration_hash() == metadata.configuration_hash()
    assert json.loads(metadata.to_json())["schema_version"] == 25
    units = transverse_units(output)
    assert (units.power, units.energy, units.intensity) == ("W/m", "J/m", "W/m^2")
    figures = stats(output)
    labels = [axis.get_ylabel() for figure in figures for axis in figure.axes]
    assert "Energy (uJ/m)" in labels
    assert any(label.startswith("Peak power (") and "W/m)" in label for label in labels)
    assert "Peak intensity (TW/cm^2)" in labels
    assert "W/m)" in time_1d(output).axes[0].get_ylabel()
    assert "/m)$^{1/2}$" in time_1d(output, quantity="field").axes[0].get_ylabel()
    assert "per m of y" in spec_1d(output).axes[0].get_ylabel()
    assert plot_energy(output).axes[0].get_ylabel() == "Energy (uJ/m)"
    plt.close("all")
    for reduction, unit in (
        (peak_power(), "W/m"),
        (energy_in_band(700e-9, 900e-9), "J/m"),
        (field_at_plane(), "sqrt(W/m)"),
    ):
        assert _provenance(reduction, metadata=metadata)["units"] == unit


@pytest.mark.medium
def test_an_area_run_carries_no_measure_key():
    """Every run that is not uniform in y keeps the record it always had."""
    simulation = Simulation.from_waveguide(
        [MarcatiliMode(HALF_GAP, cladding=1.45)],
        LENGTH,
        "Ar",
        1.0,
        **(options(energy=1e-9)),
    )
    output = simulation.run(progress="none")
    assert output.metadata is not None
    assert "transverse_measure" not in output.metadata.configuration
    assert output.metadata.transverse_measure == "area"
    assert transverse_units(output).energy == "J"


# ------------------------------------------------------------------ refusals


def test_a_slab_set_is_refused_where_it_cannot_go():
    slab = SlabMode(HALF_GAP, cladding=1.45)
    with pytest.raises(ModeError, match=r"MarcatiliMode\) measures 'area'"):
        Simulation.from_waveguide(
            [slab, MarcatiliMode(HALF_GAP)],
            LENGTH,
            "Ar",
            1.0,
            **(options()),
        )
    with pytest.raises(DriverError, match="n_theta=8 was given for a slab set"):
        Simulation.from_waveguide(
            [slab],
            LENGTH,
            "Ar",
            1.0,
            **(options(n_theta=8)),
        )
    # The general route takes a guide's own channels; a planar MaterialRegion is
    # out of scope, so a slab set beside a provider is refused.
    grid = general_grid()
    with pytest.raises(DriverError, match="mode\\(s\\) of its own"):
        Simulation.from_waveguide(
            [slab],
            GENERAL_Z_END,
            provider=general_guide(grid=grid),
            lambda0=835e-9,
            grid=grid,
        )


def test_the_default_rule_reaches_the_node_bound():
    modes = [SlabMode(HALF_GAP, cladding=1.45, order=order) for order in (1, 3, 5)]
    with pytest.warns(SolpropWarning, match="walking off"):
        simulation = Simulation.from_waveguide(
            modes,
            LENGTH,
            "Ar",
            1.0,
            **(options(radial_rule="gauss")),
        )
    rule = simulation.settings.quadrature
    assert rule is not None and rule.kind == "slab" and not rule.full
    # cartesian_nodes(5) = 36 is above default_radial_order(3) = 32.
    assert rule.n_r == 36 and rule.n_nodes == 36
    assert simulation.settings.stats is not None
    assert "quadrature_error" not in simulation.settings.stats.names


# --------------------------------------------------------- other doors, the file


@pytest.mark.medium
def test_a_slab_fibre_through_prop_gnlse_records_per_unit_width():
    """``prop_gnlse(fibre=SlabMode)`` wraps it in a SpecifiedMode, still per metre of y.

    The specified mode keeps its source in its dispersion and takes the source's
    ``aeff`` (4a/3, a width), so its powers are W/m; the measure follows that chain.
    """
    output = prop_gnlse(
        None,
        LENGTH,
        fibre=SlabMode(HALF_GAP, "Ar", 1.0),
        n2=n2("Ar", pressure=1.0, wavelength=800e-9),
        lambda0=800e-9,
        grid=GRID,
        fwhm=100e-15,
        energy=0.5,
        n_save=2,
        backend="numpy_ref",
        dz_max=LENGTH / STEPS,
        dz_init=LENGTH / STEPS,
        progress="none",
    )
    metadata = output.metadata
    assert metadata is not None and metadata.simulation_kind == "mode_averaged"
    assert metadata.transverse_measure == "per_unit_width"
    assert transverse_units(output).energy == "J/m"
    assert plot_energy(output).axes[0].get_ylabel() == "Energy (uJ/m)"
    plt.close("all")
    specified = SpecifiedMode.from_mode(SlabMode(HALF_GAP, "Ar", 1.0))
    assert transverse_measure([specified]) == "per_unit_width"
    with pytest.raises(ModeError, match="per_unit_width"):
        transverse_measure([specified, MarcatiliMode(HALF_GAP)])


class Interrupt(NullReporter):
    """Stop a run once it has saved a given number of planes."""

    def __init__(self, after):
        self.after = after
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen >= self.after:
            raise KeyboardInterrupt


@pytest.mark.medium
def test_a_slab_run_is_written_read_back_and_resumed(tmp_path):
    """The HDF5 record says sqrt(W/m) and per unit width, and a resume agrees.

    A replayed schedule at a fixed order takes the same steps and orders whatever it
    resumes from, so the planes before the interruption are a second run of the
    uninterrupted run's march, and those after it continue that march. Neither is
    promised bitwise (SEP 25) -- the gas-filled route measured 1.0e-16 of the field at
    the last plane, and a capillary through the same entry the same 1.5e-16 -- and all
    are held to the cross-backend bound of :func:`parity_bound`, which covers the same
    stage arithmetic re-formed, or the march's own budget where that is larger.
    """
    planes = [0.03, 0.06, LENGTH]

    def build():
        return Simulation.from_waveguide(
            [SlabMode(HALF_GAP, cladding=1.45)],
            LENGTH,
            "Ar",
            1.0,
            **(
                options(
                    n_save=None,
                    save_planes=planes,
                    schedule=FixedSchedule.uniform(
                        0.0, LENGTH, LENGTH / STEPS, save_planes=planes
                    ),
                    integrator=FIXED_ORDER,
                )
            ),
        )

    whole = tmp_path / "whole.h5"
    part = tmp_path / "part.h5"
    rule = build().settings.quadrature
    assert rule is not None
    transverse_nodes = rule.n_nodes
    build().run(whole, progress="none")
    with pytest.raises(KeyboardInterrupt):
        build().run(part, progress=Interrupt(2))
    with h5py.File(part, "r") as file:
        assert np.asarray(file["z"]).size < len(planes) + 1  # z = 0 is a plane too
    build().run(part, resume=True, progress="none")
    for path in (whole, part):
        with h5py.File(path, "r") as file:
            assert file.attrs["field_units"] == "sqrt(W/m)"
    with open_file(part) as reread, open_file(whole) as reference:
        assert reread.metadata is not None
        assert reread.metadata.transverse_measure == "per_unit_width"
        assert transverse_units(reread).power == "W/m"
        bound = max(parity_bound(transverse_nodes), simulation_bound(build()))
        assert_matched(reread["field"], reference["field"], bound, what="field")


@pytest.mark.medium
def test_a_slab_scan_records_its_field_units_per_unit_width(tmp_path):
    def build(point):
        return Simulation.from_waveguide(
            [SlabMode(HALF_GAP, cladding=1.45)],
            LENGTH,
            "Ar",
            float(point.values["pressure"]),
            **(options(energy=1e-9)),
        )

    path = tmp_path / "scan.h5"
    run_scan(
        Scan("slab", pressure=[1.0]),
        build,
        executor=Local(),
        output=path,
        reductions=[peak_power()],
        fields="all",
        progress="none",
    )
    found = []
    with h5py.File(path, "r") as file:
        file.visititems(
            lambda name, item: (
                found.append(json.loads(item.attrs["array_units"]))
                if "array_units" in item.attrs
                else None
            )
        )
    assert found and found[0]["fields/values"] == "sqrt(W/m)"
    assert found[0]["reduced/peak_power"] == "W/m"


def test_a_varying_gap_beside_a_gradient_says_the_fill_ignores_it():
    """Flow between plates (as a^3 per unit width) is not modelled: the uniform law."""
    gap = Linear(HALF_GAP, 50e-6, LENGTH)
    with pytest.warns(GasFlowWarning, match="slab's gap varies"):
        Simulation.from_waveguide(
            [SlabMode(gap, cladding=1.45)],
            LENGTH,
            "Ar",
            (1.0, 0.1),
            **(options()),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", GasFlowWarning)
        Simulation.from_waveguide(
            [SlabMode(gap, cladding=1.45)],
            LENGTH,
            "Ar",
            1.0,
            **(options()),
        )
        Simulation.from_waveguide(
            [SlabMode(HALF_GAP, cladding=1.45)],
            LENGTH,
            "Ar",
            (1.0, 0.1),
            **(options()),
        )

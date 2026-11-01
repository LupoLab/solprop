# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check saved-plane material processing against physical sums and file round trips."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import j0, j1

from mixture_cases import mixture_case
from solprop import fourier, processing
from solprop.constants import EPS0, C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import GasFlowWarning, ProcessingError
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.output import HDF5Output, MemoryOutput, open_file
from solprop.output.schema import SavedPlane
from solprop.physdata import Mixture
from solprop.processing._examples import example_output
from solprop.profiles import Constant, Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.translate.linop import power_weight_at


def admittance(medium, grid, transverse):
    """Return the real admittance ``c Re(k_z)/omega`` of each column and bin.

    ``k_z = sqrt((omega n / c)**2 - k_perp**2)`` from the medium's own index, on
    active positive-frequency propagating bins, zero elsewhere (SEP 34 section 4.2).
    The fill's density is constant, so the plane's position does not enter.
    """
    omega = grid.omega
    live = grid.active_mask & (omega > 0)
    eta = np.zeros((transverse.size, grid.n_omega))
    square = (omega[live] * medium.index(omega[live]) / C) ** 2 - transverse[:, None]
    root = np.sqrt(square.astype(complex))
    eta[:, live] = np.where(np.real(square) > 0, C * np.real(root) / omega[live], 0.0)
    return eta


@pytest.fixture(params=["modal", "radial", "free"])
def case(request):
    material, grid, mode, _, problem = mixture_case()
    material = Mixture.of(
        *(
            replace(
                component,
                species=replace(
                    component.species,
                    susceptibility=replace(
                        component.species.susceptibility,
                        absorption_cross_section=np.full(2, (index + 1) * 1e-25),
                    ),
                ),
            )
            for index, component in enumerate(material.components)
        )
    )
    settings = SimulationSettings(
        controller=problem.stepper.controller,
        save_planes=[grid.length],
        dz_init=1e-5,
        dz_max=1e-4,
    )
    pulse = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9)
    if request.param == "modal":
        simulation = Simulation(
            grid,
            replace(mode, mixture=material, loss=2.0),
            (),
            pulse,
            settings=settings,
        )
    else:
        space = (
            RadialGrid(6e-4, 32, window_factor=0.0)
            if request.param == "radial"
            else FreeGrid(3e-4, 4, 4e-4, 4, window_factor=0.0)
        )
        if isinstance(space, RadialGrid):
            simulation = Simulation.from_radial(
                grid,
                space,
                SpatialMedium(material),
                (),
                SpatialPulse(pulse, 80e-6),
                settings=settings,
                noise=None,
            )
        else:
            simulation = Simulation.from_free(
                grid,
                space,
                SpatialMedium(material),
                (),
                SpatialPulse(pulse, 80e-6),
                settings=settings,
                noise=None,
            )
    translated = simulation.translate()
    return simulation, translated


@pytest.fixture
def outputs(case, tmp_path):
    simulation, problem = case
    memory = MemoryOutput()
    path = tmp_path / "mixture.h5"
    file = HDF5Output(path)
    for output in (memory, file):
        output.open(simulation.metadata())
        for index, z in enumerate((0.0, simulation.grid.length)):
            output.save(SavedPlane(z=z, field=problem.initial.field0[0] * (index + 1)))
    file.close()
    with HDF5Output(path, readonly=True) as reloaded:
        yield memory, reloaded


def test_absorption_has_cross_section_density_energy_units(case, outputs):
    simulation, _ = case
    grid = simulation.grid
    # The cross-section spline and square-root spectral normalisation have fewer
    # than 32 operations; allow 128 eps per temporal FFT level and column sum.
    bound = (
        128 * (np.log2(grid.n_to) + outputs[0].fields().shape[1]) * np.finfo(float).eps
    )
    assert bound > 0
    memory, reloaded = outputs
    omega, measured = processing.absorbed_spectral_energy(memory)
    _, spectral = processing.get_spectral_density(memory, axis="omega", band="active")
    spectral = spectral.reshape(memory.n_saved, -1, omega.size)
    weights = simulation.metadata().configuration["mixture_processing"][
        "column_energy_weights"
    ]
    expected_energy = np.sum(spectral * weights[None, :, None], axis=1)
    scale = 2.0 if simulation.spatial_grid is None else 1.0
    # rho=(1,2)e24 m^-3 and sigma=(1,2)e-25 m^2 give alpha=(0.1,0.4) m^-1.
    expected = scale * np.array([0.1, 0.4])[None, :, None] * expected_energy[:, None, :]
    np.testing.assert_allclose(measured, expected, rtol=bound, atol=0)
    np.testing.assert_array_equal(
        processing.absorbed_spectral_energy(reloaded)[1], measured
    )
    np.testing.assert_array_equal(
        processing.absorbed_spectral_energy(memory, species="population-1")[1],
        measured[:, 1:2],
    )


def test_node_fluence_integrates_to_saved_energy(case, outputs):
    simulation, problem = case
    grid = simulation.grid
    # Parseval, temporal interpolation and transverse transforms cost logarithmic
    # FFT depth plus the direct QDHT reduction. The QDHT allowance of 1024 eps
    # exceeds two 32-term dot products and the spatial norm reduction. The eight-node
    # trial's 4.547e-13 bound is kept with the quadrature refined to 32 nodes: the
    # eight-node rule itself misses it, at 4.963e-10.
    node_allowance = (
        min(problem.transform.n_nodes, 8)
        if simulation.spatial_grid is not None
        and isinstance(simulation.spatial_grid, RadialGrid)
        else problem.transform.n_nodes
    )
    bound = 128 * (np.log2(grid.n_to) + node_allowance) * np.finfo(float).eps
    assert bound > 0
    memory, reloaded = outputs
    fluence = processing.node_fluence(memory)
    geometry = problem.transform.geometry
    if geometry is None:
        weights = np.array([simulation.modes[0].aeff(0.0)])
        spectral_weights = np.ones(1)
    else:
        weights = geometry.spatial_weights
        spectral_weights = EPS0 * C / 2 * geometry.spectral_weights
    spectral = fourier.spectral_density(memory.fields("fft"), grid.dt, kind=grid.kind)
    if geometry is not None:
        # Parseval over the plane leaves each column's spectral density weighted by
        # its real admittance in the fill: the signed fluence integrates to the
        # axial-flux energy (SEP 34 section 4.2).
        space = simulation.spatial_grid
        assert space is not None
        transverse = space.k_r**2 if isinstance(space, RadialGrid) else space.kperp_sq
        spectral = (
            spectral * admittance(simulation.medium, grid, transverse)[None, :, None, :]
        )
    expected = grid.d_omega * np.sum(
        spectral * spectral_weights[None, :, None, None], axis=(1, 2, 3)
    )
    np.testing.assert_allclose(fluence @ weights, expected, rtol=bound, atol=0)
    np.testing.assert_array_equal(processing.node_fluence(reloaded), fluence)


def test_saved_record_contains_ordered_species_and_complete_optics(case, outputs):
    simulation, _ = case
    memory, reloaded = outputs
    for output in (memory, reloaded):
        assert output.metadata.schema_version == 2
        material = output.metadata.configuration["medium"]
        assert material["labels"] == ["population-0", "population-1"]
        assert (
            material["temperature_K"]
            == simulation.metadata().configuration["medium"]["temperature_K"]
        )
        assert len(material["species_data"]) == 2
        if output.metadata.modes:
            assert "gas" not in output.metadata.modes[0].configuration
            assert "mixture" in output.metadata.modes[0].configuration


def test_missing_statistics_and_unknown_labels_are_explicit(outputs):
    output, _ = outputs
    for function in (
        processing.species_ionisation,
        processing.species_ionisation_energy,
    ):
        with pytest.raises(ProcessingError, match="did not record"):
            function(output)
    with pytest.raises(ProcessingError, match="misspelled"):
        processing.absorbed_spectral_energy(output, species="misspelled")


def test_legacy_record_remains_readable_without_mixture_accessors():
    output = example_output()
    for function in (
        processing.absorbed_spectral_energy,
        processing.node_fluence,
        processing.get_power_weight,
    ):
        with pytest.raises(ProcessingError, match="reconstruction data"):
            function(output)
    assert processing.get_spectral_density(output)[1].size > 0


def test_callable_total_loss_is_retained_but_species_attribution_is_refused():
    mixture, grid, mode, _, original = mixture_case()
    simulation = Simulation(
        grid,
        replace(mode, mixture=mixture, loss=np.ones_like),
        (),
        original.initial.field0[0],
        settings=SimulationSettings(
            controller=original.stepper.controller,
            save_planes=[grid.length],
            dz_init=1e-5,
            dz_max=1e-4,
        ),
    )
    output = MemoryOutput()
    output.open(simulation.metadata())
    output.save(SavedPlane(z=0.0, field=original.initial.field0[0]))
    with pytest.raises(ProcessingError, match="callable total loss"):
        processing.absorbed_spectral_energy(output)
    assert np.all(processing.node_fluence(output) >= 0)


def test_the_power_weight_is_modal_and_a_spatial_run_has_none(case):
    """Only a modal run has a weight: a spatial one already carries a physical field."""
    simulation, problem = case
    output = MemoryOutput()
    output.open(simulation.metadata())
    output.save(SavedPlane(z=simulation.grid.length, field=problem.initial.field0[0]))
    if simulation.metadata().simulation_kind in ("multimode", "mode_averaged"):
        weight = processing.get_power_weight(output)
        assert weight.shape == (1, *problem.shape[1:])
        np.testing.assert_array_equal(
            weight[0], power_weight_at(problem, simulation.grid.length)
        )
        return
    with pytest.raises(ProcessingError, match="no modal power weight"):
        processing.get_power_weight(output)


@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_unrecorded_material_positions_are_refused(case, factor):
    simulation, problem = case
    output = MemoryOutput()
    output.open(simulation.metadata())
    output.save(
        SavedPlane(z=factor * simulation.grid.length, field=problem.initial.field0[0])
    )
    with pytest.raises(ProcessingError, match="saved planes"):
        processing.node_fluence(output)


@pytest.mark.parametrize("case", ["radial"], indirect=True)
def test_a_spatial_record_without_its_plane_measure_is_refused(case):
    """A spatial record that names its model but carries no plan is not measured."""
    simulation, problem = case
    metadata = simulation.metadata()
    assert metadata.spatial is not None
    spatial = {
        key: value for key, value in metadata.spatial.items() if key != "flux_plan"
    }
    output = MemoryOutput()
    output.open(replace(metadata, spatial=spatial))
    output.save(SavedPlane(z=simulation.grid.length, field=problem.initial.field0[0]))
    with pytest.raises(ProcessingError, match="carries no plane measure"):
        processing.node_fluence(output)


def test_coarse_radial_fluence_matches_direct_bessel_synthesis():
    material, grid, _, _, original = mixture_case()
    space = RadialGrid(6e-4, 8, window_factor=0.0)
    simulation = Simulation.from_radial(
        grid,
        space,
        SpatialMedium(material),
        (),
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-9), 80e-6),
        settings=SimulationSettings(
            controller=original.stepper.controller,
            save_planes=[grid.length],
            dz_init=1e-5,
            dz_max=1e-4,
        ),
        noise=None,
    )
    output = MemoryOutput()
    output.open(simulation.metadata())
    field = simulation.translate().initial.field0[0].copy()
    # One Bessel component avoids cancellation between columns at nearly dark
    # nodes, where a componentwise relative roundoff bound is not meaningful.
    field[1:] = 0
    output.save(SavedPlane(z=0.0, field=field))
    # The inverse order-zero Bessel series uses 2/(a^2 J1(k_j a)^2).
    # This tests the physical node sum, without assuming the finite QDHT is unitary.
    bound = 128 * (np.log2(grid.n_to) + space.n_r) * np.finfo(float).eps
    assert bound > 0
    basis = (
        2
        * j0(space.r[:, None] * space.k_r[None, :])
        / (space.radius**2 * j1(space.k_r * space.radius)[None, :] ** 2)
    )
    spectrum = np.einsum("ij,jpw->ipw", basis, field)
    spectral = fourier.spectral_density(spectrum, grid.dt, kind=grid.kind)
    # One column: E and its closure share the Bessel profile, so each bin pairs to
    # |E|^2 times that column's real admittance in the fill (SEP 34 section 4.2).
    eta = admittance(SpatialMedium(material), grid, space.k_r[:1] ** 2)[0]
    expected = EPS0 * C / 2 * grid.d_omega * np.sum(spectral * eta, axis=(1, 2))
    np.testing.assert_allclose(
        processing.node_fluence(output)[0], expected, rtol=bound, atol=0
    )


def test_zero_density_does_not_require_optical_support(case):
    simulation, _ = case
    material = (
        simulation.medium.material if simulation.medium else simulation.modes[0].mixture
    )
    first, second = material.components
    table = replace(
        second.species.susceptibility,
        omega=np.array([1e16, 1e17]),
        wavelength_range=(2 * np.pi * C / 1e17, 2 * np.pi * C / 1e16),
    )
    absent = replace(
        second,
        density=Constant(0.0),
        species=replace(second.species, susceptibility=table),
    )
    mixture = Mixture.of(first, absent)
    if simulation.medium:
        simulation.medium = replace(simulation.medium, material=mixture)
    else:
        simulation.modes = tuple(
            replace(mode, mixture=mixture) for mode in simulation.modes
        )
    simulation.configuration["medium"] = mixture
    # Reconstruct a fresh simulation because the fixture has already translated.
    fresh = Simulation(
        simulation.grid,
        simulation.modes,
        (),
        simulation.initial,
        settings=simulation.settings,
        spatial_grid=simulation.spatial_grid,
        medium=simulation.medium,
        configuration=simulation.configuration,
    )
    record = fresh.metadata().configuration["mixture_processing"]
    assert isinstance(record, dict)
    np.testing.assert_array_equal(record["cross_section_m2"][1], 0.0)


@pytest.mark.parametrize("envelope", [False, True])
def test_tapered_multimode_fluence_matches_hand_node_sum(envelope):
    # No initial fraction: this run has no plasma, and a declared preionisation no
    # channel carries is refused.
    material, field_grid, mode, _, _ = mixture_case(preionisation=(0.0, 0.0))
    grid = (
        EnvelopeGrid.from_limits(field_grid.length, 800e-9, (400e-9, 2e-6), 80e-15)
        if envelope
        else field_grid
    )
    radius = Linear(30e-6, 35e-6, grid.length)
    modes = [
        replace(mode, mixture=material, radius=radius, m=index + 1)
        for index in range(2)
    ]
    simulation = Simulation.from_capillary(
        radius,
        grid.length,
        material,
        grid=grid,
        wavelength=800e-9,
        modes=modes,
        fwhm=10e-15,
        energy=1e-9,
        plasma=False,
        thg=False,
        loss=False,
        n_r=8,
        n_save=2,
        cladding=1.45,
        # No coupled flow is built on an envelope grid, so the envelope case takes
        # the adiabatic route by name.
        taper_coupling=not envelope,
        noise=None,
    )
    # A two-channel taper, which carries the taper coupling on the field grid; the
    # fluence reduction under test reads the transform, not the flow.
    problem = simulation.translate()
    output = MemoryOutput()
    output.open(simulation.metadata())
    field = problem.initial.field0[0].copy()
    field[1] = 0.5j * field[0]
    output.save(SavedPlane(z=grid.length, field=field))
    # Parseval plus two modal synthesis products and one temporal reduction.
    bound = 128 * (np.log2(grid.n_to) + 2) * np.finfo(float).eps
    assert bound > 0
    physical = np.einsum(
        "rm,mpw->rpw", problem.transform.synthesis / grid.pad_scale, field
    )
    assert problem.transform.taper_scale is not None
    expected = (
        EPS0
        * C
        / 2
        * grid.d_omega
        * np.sum(
            fourier.spectral_density(physical, grid.dt, kind=grid.kind), axis=(1, 2)
        )
        * float(problem.transform.taper_scale(grid.length)) ** 2
    )
    np.testing.assert_allclose(
        processing.node_fluence(output)[0], expected, rtol=bound, atol=0
    )


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_the_recorded_power_weight_is_factored_and_survives_the_file(tmp_path):
    # One state-sized row per save plane would make the hashed JSON record of a
    # 201-plane four-mode run 26 MB. The record holds the expansion's few terms and a
    # small per-plane basis, and a reloaded file rebuilds each plane's weight bit for
    # bit (JSON round-trips a float64, and the product is the translation's own).
    n_save = 33
    # A centimetre carrying 1.5 bar, kept at its declared ends: the flow is not
    # laminar, which the fill says.
    with pytest.warns(GasFlowWarning, match="Reynolds"):
        simulation = Simulation.from_capillary(
            125e-6,
            0.01,
            "Ar",
            (2.0, 0.5),
            vacuum_end="floor",
            wavelength=800e-9,
            wavelength_limits=(400e-9, 2e-6),
            trange=80e-15,
            modes=2,
            fwhm=10e-15,
            energy=1e-9,
            plasma=False,
            n_r=8,
            n_save=n_save,
            noise=None,
        )
    problem = simulation.translate()
    record = simulation.metadata().configuration["mixture_processing"]
    assert isinstance(record, dict)
    terms, basis = record["power_weight_terms"], record["power_weight_basis"]
    assert basis.shape == (n_save, terms.shape[0])
    assert terms.shape[0] < n_save // 4
    path = tmp_path / "run.h5"
    with HDF5Output(path) as output:
        output.open(simulation.metadata())
        for z in (0.0, *problem.stepper.save_planes[:2]):
            output.save(SavedPlane(z=float(z), field=problem.initial.field0[0]))
    with open_file(path) as reloaded:
        weights = processing.get_power_weight(reloaded)
        assert weights.shape == (3, *problem.shape[1:])
        for weight, z in zip(weights, reloaded["z"], strict=True):
            np.testing.assert_array_equal(weight, power_weight_at(problem, float(z)))

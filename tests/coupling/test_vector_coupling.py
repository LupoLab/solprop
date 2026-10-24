# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Coupling into a general modal guide: the vector model and the flux recovery.

SEP 31 section 4.3. ``vector_paraxial`` recovers a beam's coordinates in a guide
that carries E/H at fixed nodes with the flux pairing F; a radial and a Cartesian
plane of one beam launch the same amplitudes, an ordinary modal plane enters
through its synthesised field, and every unqualified pair is refused by name.
``same_basis`` recovers a general plane read with its simulation in another basis
at the same nodes by ``a = F^-1 B``. The guide is the weakly guiding 4 um strand
of the incident-launch tests; a propagation is two steps.
"""

import dataclasses
import math
from typing import cast

import numpy as np
import pytest

from modal_provider_cases import GENERAL_Z_END, general_grid, general_guide
from solprop.constants import C
from solprop.coupling import (
    VECTOR_STEP_INDEX_ENVELOPE,
    CouplingModel,
    FieldPlane,
    PlaneBasis,
    PlaneGeometry,
    ScalarModeBasis,
    couple,
)
from solprop.driver import Simulation
from solprop.exceptions import CouplingError
from solprop.grid import FreeGrid, RadialGrid
from solprop.modes import StepIndexMode
from solprop.processing import field_plane
from solprop.pulses import GaussianPulse, SpatialPulse

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GeneralModalBasisWarning"
)
"""The strand's nine knots depart from orthonormal near the band's long edge, where
the launched spectrum puts nothing (``tests/driver/test_general_modal.py``)."""

PARITIES = (("HE", 1, 1, "even"), ("HE", 1, 1, "odd"))
BUDGET = 0.2
VECTOR = CouplingModel("vector_paraxial", model_error=BUDGET)
SAME = CouplingModel("same_basis")
WAVELENGTH, FWHM, ENERGY = 835e-9, 20e-15, 1e-12


def strand(grid, descriptors=PARITIES[:1], blocks=(), n_r=24, circular=None):
    """Return a guide of a 4 um core at a 0.01 index step, nine knots.

    The step is below the vector model's contrast screen. ``circular`` declares
    the circular partners of a degenerate pair as the basis.
    """
    return general_guide(
        descriptors=descriptors,
        blocks=blocks,
        radius=4e-6,
        grid=grid,
        n_r=n_r,
        knots=9,
        indices=(1.45, 1.44),
        circular=circular,
    )


@pytest.fixture(scope="module")
def grid():
    """The general modal tests' field grid."""
    return general_grid()


def simulation(guide, grid):
    """Return a short general modal simulation of a guide on the grid."""
    return Simulation.from_waveguide(
        (),
        GENERAL_Z_END / 10,
        provider=guide,
        grid=grid,
        lambda0=WAVELENGTH,
        energy=ENERGY,
        fwhm=FWHM,
        pulseshape="gauss",
        backend="numpy_ref",
        dz_max=GENERAL_Z_END / 20,
        dz_init=GENERAL_Z_END / 20,
        n_save=2,
    )


@pytest.fixture(scope="module")
def pair(grid):
    """The strand's two HE11 parities, declared degenerate, as a simulation."""
    guide = strand(grid, PARITIES, ((0, 1),))
    return simulation(guide, grid)


@pytest.fixture(scope="module")
def pair_port(pair):
    """The pair's input port."""
    return pair.input_port()


def beam(waist):
    """Return an x-polarised Gaussian beam carrying the tests' pulse."""
    return SpatialPulse(
        GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=ENERGY), waist
    )


def step_index_plane(grid, jones):
    """Return an ordinary modal plane of the strand's LP01 profile.

    ``jones`` gives the x and, for two polarisations, the y amplitude; the spectrum
    is the tests' pulse's Gaussian.
    """
    profiles = ScalarModeBasis.from_modes(
        [StepIndexMode(4e-6, core=1.45, cladding=1.44, profile_wavelength=WAVELENGTH)],
        grid,
    )
    width = FWHM / (2 * math.sqrt(math.log(2)))
    envelope = np.exp(-(((grid.omega - grid.omega0) * width) ** 2) / 2)
    samples = np.zeros((1, len(jones), grid.n_omega), dtype=complex)
    for p, amplitude in enumerate(jones):
        samples[0, p] = amplitude * envelope * grid.active_mask
    return FieldPlane.from_samples(
        samples,
        grid,
        PlaneGeometry(0.0),
        exterior="zero",
        basis=PlaneBasis("modal", "0" * 64, profiles.labels),
        profiles=profiles,
    )


def channel_energy(result):
    """Return each launched channel's energy, J (identity metric)."""
    return np.sum(
        result.launch.samples[:, 0, :].real ** 2
        + result.launch.samples[:, 0, :].imag ** 2,
        axis=1,
    )


def test_a_cartesian_plane_launches_what_the_radial_plane_of_the_beam_does(
    grid, pair_port
):
    """One beam on a radial and on a Cartesian plane gives one launch.

    Both planes resolve the 6 um beam to rounding, so the field each gives at the
    guide's nodes is the same and so are the recovered amplitudes. The Cartesian
    window's half-width, 25 um, is inside the guide's outermost node: the nodes
    beyond it see the declared zero exterior, and the beam's share near the window's
    edge, counted as numerical error, is rounding.
    """
    index = np.asarray(pair_port.reference_index)
    radial = FieldPlane.from_pulse(
        beam(6e-6),
        grid,
        PlaneGeometry(0.0),
        space=RadialGrid(60e-6, 128),
        index=index,
    )
    cartesian = FieldPlane.from_pulse(
        beam(6e-6),
        grid,
        PlaneGeometry(0.0),
        space=FreeGrid(25e-6, 64, 25e-6, 64),
        index=index,
    )
    assert cartesian.exterior == "zero"
    assert np.max(np.hypot(*np.asarray(pair_port.nodes).T)) > 25e-6
    first = couple(radial, pair_port, model=VECTOR)
    second = couple(cartesian, pair_port, model=VECTOR)
    scale = float(np.max(np.abs(first.launch.samples)))
    np.testing.assert_allclose(
        second.launch.samples, first.launch.samples, rtol=0.0, atol=1e-9 * scale
    )
    numerical = next(i for i in second.report.items if i.name == "numerical")
    assert numerical.value is not None and numerical.value < 1e-12
    assert 0.0 < second.report.launched_energy < second.report.incident_energy


def test_an_ordinary_modal_plane_enters_through_its_synthesised_field(grid, pair_port):
    """A step-index LP01 plane fills the strand's HE11 to its weak-guidance overlap.

    The LP01 profile is the weak-guidance limit of HE11, so nearly all of it is
    launched. With a y component of half the x amplitude, the even channel takes
    the x field and the odd one the y field, a quarter of the energy.
    """
    single = couple(step_index_plane(grid, (1.0,)), pair_port, model=VECTOR)
    report = single.report
    assert 0.999 < report.launched_energy / report.incident_energy < 1.0
    assert report.budget <= BUDGET
    both = couple(step_index_plane(grid, (1.0, 0.5j)), pair_port, model=VECTOR)
    assert both.report.incident_energy == pytest.approx(
        1.25 * report.incident_energy, rel=1e-12
    )
    even, odd = channel_energy(both)
    assert odd / even == pytest.approx(0.25, rel=1e-6)


def test_a_plane_with_no_field_is_a_valid_zero_launch(grid, pair_port):
    """Nothing is occupied, so nothing is probed and nothing is launched."""
    empty = dataclasses.replace(
        step_index_plane(grid, (1.0,)),
        samples=np.zeros((1, 1, grid.n_omega), dtype=complex),
    )
    result = couple(empty, pair_port, model=VECTOR)
    assert result.report.incident_energy == 0.0
    assert result.report.launched_energy == 0.0
    assert not np.any(result.launch.samples)


def test_vector_paraxial_refuses_unqualified_pairs_by_name(grid, pair_port):
    """A port without nodal E/H, a modal plane without profiles, a general plane, a
    window short of the nodes with an unknown exterior, and evanescent columns."""
    plane = step_index_plane(grid, (1.0,))
    for port in (
        dataclasses.replace(pair_port, electric=None),
        dataclasses.replace(
            pair_port, route="modal", basis=PlaneBasis("modal", "0" * 64)
        ),
    ):
        with pytest.raises(CouplingError, match="this port does not"):
            couple(plane, port, model=VECTOR)
    with pytest.raises(CouplingError, match="not a modal plane without profiles"):
        couple(dataclasses.replace(plane, profiles=None), pair_port, model=VECTOR)
    general = FieldPlane.from_samples(
        np.zeros((2, 1, grid.n_omega), dtype=complex),
        grid,
        PlaneGeometry(0.0),
        exterior="zero",
        basis=PlaneBasis("general_modal", "0" * 64),
    )
    with pytest.raises(CouplingError, match=r"not a general_modal plane$"):
        couple(general, pair_port, model=VECTOR)
    index = np.asarray(pair_port.reference_index)

    def gaussian(waist):
        def field(r, omega):
            # Narrow enough to vanish to rounding at the active band's edges.
            spectrum = np.exp(-(((omega - grid.omega0) * FWHM) ** 2))
            return np.exp(-((r / waist) ** 2))[:, None] * spectrum[None, :]

        return field

    cropped = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        gaussian(6e-6),
        space=RadialGrid(30e-6, 128),
        index=index,
        exterior="unknown",
    )
    with pytest.raises(CouplingError, match="exterior is unknown"):
        couple(cropped, pair_port, model=VECTOR)
    # A 0.2 um waist has angular content far beyond n omega / c.
    evanescent = FieldPlane.from_analytic(
        grid,
        PlaneGeometry(0.0),
        gaussian(0.2e-6),
        space=RadialGrid(10e-6, 256),
        index=index,
    )
    with pytest.raises(CouplingError, match="columns that do not propagate"):
        couple(evanescent, pair_port, model=VECTOR)


@pytest.mark.medium
def test_a_general_plane_is_recovered_in_another_basis_at_common_nodes(grid, pair):
    """``same_basis`` recovers a general plane through F^-1 B, or names why not.

    The pair is launched in its even channel and run two steps; its exit plane,
    read with its simulation, carries E and H at the strand's nodes. Recovered in
    the even channel alone (another basis at the same nodes) it keeps its energy to
    rounding: the odd channel it leaves out is empty. Recovered in the same pair's
    dense circular basis it keeps its energy to the plane's own metric defect: the
    pair's identity metric departs from its flux at the band's long edge, and the
    span check allows that defect as well as the port's. The odd channel alone does
    not span it; a plane read without its simulation has no E/H; a port without
    them, or at other nodes, cannot pair it.
    """
    output = pair.run(progress="none")
    plane = field_plane(output, source=pair)
    assert plane.nodal is not None
    even = simulation(strand(grid), grid).input_port()
    assert even.basis.identity != plane.basis.identity
    result = couple(plane, even, model=SAME)
    assert result.report.launched_energy == pytest.approx(
        result.report.incident_energy, rel=1e-10
    )
    circular = simulation(strand(grid, PARITIES, ((0, 1),), circular=(0, 1)), grid)
    circular_port = circular.input_port()
    assert circular_port.metric.kind == "dense"
    recovered = couple(plane, circular_port, model=SAME).report
    items = {item.name: item for item in recovered.items}
    source_defect = items["source metric"].value
    assert source_defect is not None
    # Measured 1.1e-6: the pair's identity metric against its flux, weighted by
    # the plane's spectrum; the dense circular metric is the port's own F.
    assert 1e-8 < source_defect < 1e-5
    assert items["target metric"].value == pytest.approx(0.0, abs=1e-12)
    assert abs(recovered.launched_energy / recovered.incident_energy - 1.0) <= (
        source_defect + 1e-10
    )
    odd_guide = strand(grid, PARITIES[1:])
    with pytest.raises(CouplingError, match="not in the port's span"):
        couple(plane, simulation(odd_guide, grid).input_port(), model=SAME)
    with pytest.raises(CouplingError, match="read it with source="):
        couple(field_plane(output), even, model=SAME)
    with pytest.raises(CouplingError, match="the port's E/H at fixed nodes"):
        couple(plane, dataclasses.replace(even, magnetic=None), model=SAME)
    coarse = strand(grid, n_r=20)
    with pytest.raises(CouplingError, match="common quadrature"):
        couple(plane, simulation(coarse, grid).input_port(), model=SAME)


def envelope_item(report):
    """Return a report's envelope item."""
    return next(item for item in report.items if item.name == "envelope")


def test_a_step_index_guide_counts_its_recorded_facet_envelope(grid, pair_port):
    """A StepIndexProvider port carries V; the facet envelope's allowance is counted.

    The strand is V = 5.1 at 835 nm with a 0.0069 contrast, inside the recorded
    range; the item is the bound scaled by the weighted squared contrast, plus the
    incident share of any bins whose V leaves the range.
    """
    frequency = np.asarray(pair_port.normalised_frequency)
    centre = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    # V = (omega/c) a sqrt(1.45^2 - 1.44^2) for the 4 um strand.
    expected = grid.omega[centre] / C * 4e-6 * math.sqrt(1.45**2 - 1.44**2)
    assert frequency[centre] == pytest.approx(expected, rel=1e-12)
    report = couple(step_index_plane(grid, (1.0,)), pair_port, model=VECTOR).report
    item = envelope_item(report)
    assert item.counted
    assert item.note.startswith(VECTOR_STEP_INDEX_ENVELOPE.identity)
    incident = report.incident / np.sum(report.incident)
    contrast = np.asarray(pair_port.contrast)
    low, high = VECTOR_STEP_INDEX_ENVELOPE.ranges["v"]
    beyond = float(np.sum(incident[(frequency < low) | (frequency > high)]))
    allowance = VECTOR_STEP_INDEX_ENVELOPE.allowance(
        {"contrast_squared": float(np.sum(contrast**2 * incident))}
    )
    assert item.value == pytest.approx(allowance + beyond, rel=1e-12)
    assert report.budget <= BUDGET


def test_bins_beyond_the_facet_envelope_add_their_incident_share(grid, pair_port):
    """Bins whose V leaves the range are counted in full, and the item says so."""
    frequency = np.asarray(pair_port.normalised_frequency)
    centre = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    # Move the centre bin alone to V = 8.2, just beyond the range.
    stretched = frequency.copy()
    stretched[centre] = 8.2
    port = dataclasses.replace(pair_port, normalised_frequency=stretched)
    model = CouplingModel("vector_paraxial", model_error=0.9)
    report = couple(step_index_plane(grid, (1.0,)), port, model=model).report
    share = report.incident[centre] / np.sum(report.incident)
    item = envelope_item(report)
    assert "incident share outside its range" in item.note
    within = envelope_item(
        couple(step_index_plane(grid, (1.0,)), pair_port, model=VECTOR).report
    )
    assert cast("float", item.value) == pytest.approx(
        cast("float", within.value) + share, rel=1e-6
    )


@pytest.mark.parametrize(
    ("change", "named"),
    [({"frequency": 2.0}, "v = "), ({"contrast": 8.0}, "contrast = ")],
)
def test_a_step_index_guide_outside_the_facet_envelope_is_refused(
    grid, pair_port, change, named
):
    """A guide whose weighted V or contrast leaves the range is refused by name."""
    port = dataclasses.replace(
        pair_port,
        normalised_frequency=np.asarray(pair_port.normalised_frequency)
        * change.get("frequency", 1.0),
        contrast=np.asarray(pair_port.contrast) * change.get("contrast", 1.0),
    )
    model = CouplingModel("vector_paraxial", model_error=0.9)
    with pytest.raises(
        CouplingError, match=VECTOR_STEP_INDEX_ENVELOPE.identity
    ) as error:
        couple(step_index_plane(grid, (1.0,)), port, model=model)
    assert named in str(error.value)
    assert "not qualified for this step-index guide" in str(error.value)


def test_a_guide_without_a_recorded_envelope_keeps_an_uncounted_item(grid, pair_port):
    """A general guide that is not step-index (a graded index) carries no V; its
    envelope item has no value and says why, as before the envelope was recorded."""
    port = dataclasses.replace(pair_port, normalised_frequency=None)
    report = couple(step_index_plane(grid, (1.0,)), port, model=VECTOR).report
    item = envelope_item(report)
    assert item.value is None
    assert not item.counted
    assert "only a step-index guide has one" in item.note


def test_a_beam_in_another_medium_than_the_core_is_refused(grid, pair_port):
    """The facet envelope covers a beam in the core's index only.

    A beam in air meeting a 1.45 core reflects 3 per cent at the bulk step alone
    (the facet reference puts the model 6 per cent high there), which is a separate
    planar interface (SEP 31, section 4.3). The model measures the beam in the
    core's admittance, so the source metric item, the beam's own measure against
    that, refuses it.
    """
    plane = FieldPlane.from_pulse(
        beam(6e-6),
        grid,
        PlaneGeometry(0.0),
        space=RadialGrid(60e-6, 128),
        index=np.where(np.asarray(pair_port.reference_index) > 0, 1.0, 0.0),
    )
    with pytest.raises(CouplingError, match="source metric"):
        couple(plane, pair_port, model=VECTOR)

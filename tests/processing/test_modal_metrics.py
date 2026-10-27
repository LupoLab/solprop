# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physical E and H on a saved plane, and the three measures they distinguish.

What is tested here: the reconstruction round trip against the coefficient dual, the
distinction between the physical energy, the cycle-averaged flux, the instantaneous flux
and the per-coordinate intensity, the labels a schema-7 file carries, and the refusals
-- an old file asked for a general name, a general file asked for a legacy one, and a
reconstruction offered the wrong guide's tables.

**Every bound is written before the run.** The reconstruction is one matrix product per
node and one transform, so its round trip against the dual is a rounding statement whose
budget is the reduction length; the flux identities are exact arithmetic identities and
are bounded the same way. Nothing here propagates: one saved plane of a run of ten steps
over a tenth of a millimetre is all a reconstruction needs.

**A nonunitary or frequency-dependent metric is refused, not measured.** A file records
the metric's identity and not its arrays, so the metric-weighted accessors admit the
identity metric alone, and what this module can test of any other metric is the
refusal.
"""

import dataclasses
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

from modal_provider_cases import GENERAL_Z_END, general_grid, general_guide
from solprop import stats
from solprop.backends import select
from solprop.driver import Simulation
from solprop.exceptions import BackendError, OutputError
from solprop.grid import FieldGrid
from solprop.output.capability import (
    GENERAL_MODAL_ACCESSORS,
    capability_of,
    declared_capability,
    refusal_for,
)
from solprop.output.hdf5 import open_file
from solprop.output.memory import MemoryOutput
from solprop.processing import time as legacy
from solprop.processing._examples import example_general_output, example_output
from solprop.processing.axes import grid_of
from solprop.processing.modal import (
    channel_amplitudes,
    channel_labels,
    coordinate_intensity,
    electric_field,
    instantaneous_flux,
    magnetic_field,
    physical_energy,
    plane_nodes,
    reconstructed_flux,
)
from solprop.stats import StatsSpec
from solprop.translate.general import general_modal_route

PULSE = {"lambda0": 835e-9, "fwhm": 20e-15, "energy": 1e-12, "pulseshape": "gauss"}
RUN = {
    # Two steps over a tenth of a millimetre: a reconstruction is made on one
    # saved plane and needs no propagation at all, and the route's arithmetic is
    # qualified end to end in tests/driver/test_general_modal.py.
    "save_planes": [GENERAL_Z_END / 10],
    "backend": "numpy_ref",
    "dz_max": GENERAL_Z_END / 20,
    "dz_init": GENERAL_Z_END / 20,
}

ROUNDING = 1e-12
"""Budget for an identity that is arithmetic, not physics.

The reconstruction contracts the stored reference over the channel axis and the recovery
contracts the dual over nodes and six components, so a round trip is of order
``n_nodes`` roundings of numbers of one magnitude: 288 nodes at 2.2e-16 is 6.4e-14, and
1e-12 is the same order of cover every other rounding statement in this module takes.
"""


@pytest.fixture(scope="module")
def grid():
    """The field grid these runs are on."""
    return general_grid()


@pytest.fixture(scope="module")
def guide(grid):
    """One HE11 channel of a 1.5 um strand, sampled over the run's active band."""
    return general_guide(grid=grid)


@pytest.fixture(scope="module")
def pair_guide(grid):
    """The two parities of that channel, declared degenerate."""
    return general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        blocks=((0, 1),),
        grid=grid,
    )


def route_for(guide, grid):
    """Return the route a run of these tests builds, rebuilt from the guide.

    `Simulation._from_general` passes the **last save plane** as `z_max`, so a
    caller rebuilding the reference for a reconstruction passes the same: the
    tables' `z_span` is part of their identity and a different one is refused by
    name, which is the point of the check.
    """
    return general_modal_route(
        guide,
        grid,
        z_max=float(np.max(RUN["save_planes"])),
        omega_reference=float(grid.omega0),
    )


@pytest.fixture(scope="module")
def tables(guide, grid):
    """The run's own reference, as `general_modal_route` builds it."""
    return route_for(guide, grid).tables


@pytest.fixture(scope="module")
def run(guide, grid):
    """One general modal run, in memory, with its two saved planes."""
    simulation = Simulation.from_waveguide(
        (), GENERAL_Z_END, provider=guide, grid=grid, **(PULSE | RUN)
    )
    return cast("MemoryOutput", simulation.run(progress="none"))


# ------------------------------------------------------------------ reconstruction


def test_the_reconstruction_carries_si_units_and_three_components(
    run, tables, grid, guide
):
    """E in V/m and H in A/m at every node, with a longitudinal component."""
    electric = electric_field(run, tables, plane=-1, guide=guide)
    magnetic = magnetic_field(run, tables, plane=-1, guide=guide)
    assert electric.shape == (tables.node_xy.shape[1], 3, grid.n_t)
    assert magnetic.shape == electric.shape
    assert np.all(np.isfinite(electric)) and np.all(np.isfinite(magnetic))
    # A bound HE11 mode of a strand is mostly transverse, and its longitudinal
    # component is small but not zero: that is the three-component physics this
    # route exists for, and a zero here would mean the reference was not read.
    transverse = float(np.max(np.abs(electric[:, :2])))
    assert 0.0 < float(np.max(np.abs(electric[:, 2]))) < transverse
    assert np.array_equal(
        np.real(electric_field(run, tables, plane=-1, analytic=True, guide=guide)),
        electric,
    )


def test_the_coefficient_dual_recovers_the_amplitudes_it_synthesised(
    run, tables, grid, record_property
):
    """The round trip of A2 companion section 4, to the rounding budget above."""
    stored = np.sum(run.fields("fft")[-1], axis=1)
    reference = tables.electric[0]
    magnetic = tables.magnetic[0]
    electric = np.einsum("qcmw,mw->qcw", reference, stored)
    fields = np.einsum("qcmw,mw->qcw", magnetic, stored)
    recovered = channel_amplitudes(electric, fields, tables)
    scale = float(np.max(np.abs(stored)))
    residual = float(np.max(np.abs(recovered - stored))) / scale
    record_property("dual_round_trip", residual)
    assert residual < ROUNDING
    # And it is the frequency axis of the run, not of the guide's knots.
    assert recovered.shape == (tables.electric.shape[3], grid.n_omega)


def test_a_dual_that_is_absent_or_mismatched_is_refused_by_name(tables):
    """Defence in depth: the dual is optional on the record and is checked."""
    without = dataclasses.replace(tables, coordinate_dual=None)
    shape = (tables.node_xy.shape[1], 3, len(tables.omega))
    zeros = np.zeros(shape, complex)
    with pytest.raises(OutputError, match="carry no coordinate_dual"):
        channel_amplitudes(zeros, zeros, without)
    with pytest.raises(OutputError, match="the nodal fields have shapes"):
        channel_amplitudes(zeros[:, :, :-1], zeros[:, :, :-1], tables)


# -------------------------------------------------------- energy, flux, intensity


def _cross_frequency_bound(run, tables):
    """Return a rigorous bound on how far the reconstructed peak may sit.

    The reconstructed flux is
    ``Re sum_ij K_ij B_i conj(B_j) exp(-i (w_i - w_j) t)`` with
    ``K_ij = sum_q w_q [e_x(w_i) conj(h_y(w_j)) - e_y(w_i) conj(h_x(w_j))] / 2``
    the guide's **cross-frequency** flux kernel, while the statistic is
    ``max_t |A(t)|^2``, the same double sum with every coefficient one: the
    provider normalises the same-frequency flux to unity and says nothing about
    ``K_ij`` off the diagonal. The difference is therefore bounded, term by term,
    by ``sum_ij |B_i| |B_j| |K_ij - 1|``, which is what this returns, relative to
    the statistic's own peak. Nothing here reads the reconstruction.

    Parameters
    ----------
    run : solprop.output.memory.MemoryOutput
        The run, whose last plane's amplitudes and statistic are read.
    tables : solprop.translate.modal_tables.ModalTransformTables
        The guide's stored reference.

    Returns
    -------
    tuple of float
        The relative bound and the largest off-diagonal departure of ``K``.
    """
    grid = grid_of(run)
    amplitudes = np.sum(run.fields("fft")[-1], axis=1)[0]
    electric = tables.electric[0, :, :, 0, :]
    magnetic = tables.magnetic[0, :, :, 0, :]
    weights = tables.weights[0]
    kernel = 0.5 * (
        np.einsum("q,qi,qj->ij", weights, electric[:, 0], np.conj(magnetic[:, 1]))
        - np.einsum("q,qi,qj->ij", weights, electric[:, 1], np.conj(magnetic[:, 0]))
    )
    # `analytic_field` applies the doubling and the transform's own scale; the
    # one-hot probe measures both at once, so that the bound is in the units the
    # statistic is in without guessing a normalisation.
    probe = np.max(
        np.abs(stats.analytic_field(grid, np.eye(grid.n_omega, dtype=complex))),
        axis=-1,
    )
    magnitude = np.abs(amplitudes) * probe
    departure = np.abs(kernel - 1.0)
    numerator = float(magnitude @ departure @ magnitude)
    recorded = float(np.ravel(run.planes()[-1].stats["analytic_flux"])[0])
    # Outside the guide's support the stored reference is an explicit zero, and so
    # is the state there; the diagonal is one only where the guide is solved.
    support = np.any(np.abs(electric) > 0.0, axis=(0, 1))
    diagonal = float(np.max(np.abs(np.diag(kernel)[support] - 1.0)))
    return numerator / recorded, diagonal


def test_the_reconstructed_peak_is_the_statistic_plus_the_cross_frequency_term(
    run, tables, guide, record_property
):
    """The distinction A2 companion section 6 insists on, measured.

    The run statistic ``analytic_flux`` is the peak of the **same-frequency**
    diagonal, which is what a modal state can say on its own; the reconstruction
    integrates the physical ``Re(E_a x conj(H_a))/2`` over the cross-section and
    therefore carries the cross-frequency flux of the guide's own profiles. Even a
    positive spectral metric does not make an instantaneous cross-frequency flux a
    sum of coordinate powers, and here the difference is measurable.

    The bound is the term-by-term triangle inequality on that difference, built
    from the guide's kernel and the launched spectrum **before** the
    reconstruction is read.
    """
    bound, diagonal = _cross_frequency_bound(run, tables)
    record_property("cross_frequency_bound", bound)
    record_property("kernel_diagonal_departure", diagonal)
    # The provider normalises the same-frequency flux to one at every knot, so
    # the diagonal departs from one by what the table's frequency interpolation
    # costs, which the guide measured and the record carries.
    profile_error = float(run.metadata.modal["guide"]["sampling"]["profile_error"])
    record_property("profile_error", profile_error)
    assert diagonal < profile_error
    recorded = float(np.ravel(run.planes()[-1].stats["analytic_flux"])[0])
    reconstructed = float(
        np.max(reconstructed_flux(run, tables, plane=-1, guide=guide))
    )
    departure = abs(reconstructed - recorded) / recorded
    record_property("flux_against_the_statistic", departure)
    assert departure < bound + ROUNDING
    # And it is not zero: this is the term the two accessors differ by.
    assert departure > 0.0


def test_a_plane_s_nodes_are_where_the_accessors_return_their_values(
    run, tables, guide
):
    """The nodes and weights a constant guide's plane is reconstructed on.

    They are the tables' own at every plane, and the flux density the accessor
    returns at them, integrated with these weights, is the accessor's own flux: the
    same ``einsum`` over the same weights, so the two agree to the rounding of a
    reduction over the node axis.
    """
    xy, weights = plane_nodes(run, tables, plane=-1, guide=guide)
    np.testing.assert_array_equal(xy, tables.node_xy[0])
    np.testing.assert_array_equal(weights, tables.weights[0])
    assert xy.shape == (weights.size, 2)
    first, _ = plane_nodes(run, tables, plane=0, guide=guide)
    np.testing.assert_array_equal(first, xy)
    density = reconstructed_flux(run, tables, plane=-1, density=True, guide=guide)
    assert density.shape[0] == weights.size
    flux = reconstructed_flux(run, tables, plane=-1, guide=guide)
    assert_allclose(weights @ density, flux, rtol=0, atol=ROUNDING * np.max(flux))


def test_the_energy_is_the_time_integral_of_the_flux(
    run, tables, guide, record_property
):
    """Parseval, through the physical fields: `int P dt` is the physical energy.

    Exact where the peak is not: the cross-frequency terms oscillate at an integer
    number of cycles in the window and integrate to nothing, so the time integral
    of the reconstructed flux is the same-frequency sum the statistic takes. What
    is left is the sampled Gram defect of the basis, and arithmetic.
    """
    grid = grid_of(run)
    flux = reconstructed_flux(run, tables, plane=-1, guide=guide)
    integrated = float(np.sum(flux)) * float(grid.dt)
    recorded = float(np.ravel(run.planes()[-1].stats["physical_energy"])[0])
    departure = abs(integrated - recorded) / recorded
    record_property("energy_against_the_flux_integral", departure)
    _, diagonal = _cross_frequency_bound(run, tables)
    assert departure < diagonal + ROUNDING


def test_the_instantaneous_flux_averages_to_the_analytic_one(
    run, tables, guide, record_property
):
    """The sum-frequency term integrates to nothing over the window.

    ``E H`` of the real fields is ``Re(E_a conj(H_a))/2`` plus a term at twice the
    carrier. The second term is what distinguishes the two accessors and what a
    cycle-averaging detector does not see; over the whole window it sums to a
    fraction of the first, set by how many carrier cycles the window holds.
    """
    analytic = reconstructed_flux(run, tables, plane=-1, guide=guide)
    instantaneous = instantaneous_flux(run, tables, plane=-1, guide=guide)
    assert instantaneous.shape == analytic.shape
    # They are not the same trace: the sum-frequency term is of the same size as
    # the flux itself at any one instant.
    peak = float(np.max(np.abs(analytic)))
    excursion = float(np.max(np.abs(instantaneous - analytic))) / peak
    record_property("sum_frequency_excursion", excursion)
    assert excursion > 0.1
    # And their integrals agree: the oscillation averages away over the window.
    departure = abs(float(np.sum(instantaneous - analytic))) / float(np.sum(analytic))
    record_property("instantaneous_against_analytic_integral", departure)
    assert departure < ROUNDING


def test_a_coordinate_intensity_is_not_the_physical_one(pair_guide, grid):
    """Two channels launched together interfere, and coordinate powers do not add.

    The distinction A2 companion section 6 insists on: the sum over channels of
    the per-coordinate flux densities is the physical density only where the
    channels do not interfere. Here they do, at the same nodes, so the difference
    is a measurable fraction of the total and is **not** clipped away.
    """
    route = route_for(pair_guide, grid)
    simulation = Simulation.from_waveguide(
        (), GENERAL_Z_END, provider=pair_guide, grid=grid, **(PULSE | RUN)
    )
    built = np.asarray(simulation.initial)
    # Both parities at once, at equal amplitude: a linearly polarised launch at
    # 45 degrees, which no single coordinate carries.
    state = np.zeros_like(built)
    state[0] = built[0] / np.sqrt(2.0)
    state[1] = built[0] / np.sqrt(2.0)
    output = cast("MemoryOutput", simulation.with_input(state).run(progress="none"))
    per_channel = coordinate_intensity(output, route.tables, plane=-1, guide=pair_guide)
    total = reconstructed_flux(
        output, route.tables, plane=-1, density=True, guide=pair_guide
    )
    assert per_channel.shape == (2, *total.shape)
    scale = float(np.max(np.abs(total)))
    interference = float(np.max(np.abs(np.sum(per_channel, axis=0) - total))) / scale
    assert interference > 1e-3
    # Nothing is clipped: the interference is signed and reaches both ways.
    difference = np.sum(per_channel, axis=0) - total
    assert float(np.min(difference)) < 0.0 < float(np.max(difference))


def test_the_file_based_energy_is_the_statistic_the_run_recorded(run):
    """`processing.modal.physical_energy` calls the one definition, so it agrees."""
    recorded = np.array(
        [float(np.ravel(plane.stats["physical_energy"])[0]) for plane in run.planes()]
    )
    assert_allclose(physical_energy(run), recorded, rtol=0.0, atol=0.0)
    assert float(physical_energy(run, plane=-1)) == recorded[-1]


def test_the_two_statistics_are_call_throughs_to_their_legacy_arithmetic(run):
    """A regression guard, not a measurement: the two are literal call-throughs.

    `stats.physical_energy` calls `stats.energy` and sums it, and
    `stats.analytic_flux` **is** `stats.peak_power_total`, so the equality below
    is bitwise by construction rather than by agreement of two computations. It
    is asserted because the docstrings promise it and because giving either name a
    metric argument is where it would stop being true.
    """
    field = run.fields("fft")[-1]
    live = grid_of(run)
    assert float(stats.physical_energy(live, field)) == float(
        np.sum(stats.energy(live, field))
    )
    assert float(stats.analytic_flux(live, field)) == float(
        stats.peak_power_total(live, field)
    )


# ------------------------------------------------------------ labels and refusals


def test_a_schema_seven_file_names_its_channels(run):
    """The labels persisted on the mode axis come back as labels."""
    labels = channel_labels(run)
    assert [one.display_label for one in labels] == ["HE1,1:even"]
    assert labels[0].family == "HE"
    assert (labels[0].azimuthal_order, labels[0].radial_order) == (1, 1)
    assert [dict(one.as_mapping()) for one in labels] == list(
        run.metadata.modal["channels"]
    )


def test_an_older_file_has_no_labels_and_says_so():
    """A run whose records were built before schema 7 names no channel."""
    with pytest.raises(OutputError, match="carries no channel labels"):
        channel_labels(example_output())


def test_an_ordinary_run_is_refused_every_general_accessor(tables, guide):
    """The reviewer's trap: an old file asked for a reconstruction."""
    ordinary = example_output()
    for accessor in (
        electric_field,
        magnetic_field,
        reconstructed_flux,
        instantaneous_flux,
        plane_nodes,
    ):
        with pytest.raises(OutputError, match="is unavailable for this modal run"):
            accessor(ordinary, tables, plane=-1, guide=guide)
    with pytest.raises(OutputError, match="is unavailable for this modal run"):
        coordinate_intensity(ordinary, tables, plane=-1, guide=guide)
    with pytest.raises(OutputError, match="is unavailable for this modal run"):
        physical_energy(ordinary)


def test_a_handler_with_no_record_is_refused(tables, guide):
    """Nothing is reconstructed from a handler that was never opened."""
    empty = MemoryOutput()
    with pytest.raises(OutputError, match="needs an opened output"):
        electric_field(empty, tables, plane=-1, guide=guide)
    with pytest.raises(OutputError, match="needs an opened output"):
        channel_labels(empty)


def test_a_reconstruction_needs_one_plane(run, tables, guide):
    """A whole run of physical fields is not what an accessor returns by default."""
    with pytest.raises(OutputError, match="made on one saved plane"):
        electric_field(run, tables, guide=guide)
    with pytest.raises(OutputError, match="made on one saved plane"):
        plane_nodes(run, tables, guide=guide)


def test_the_wrong_guides_tables_are_refused_by_the_field_that_disagrees(
    run, grid, tables, guide
):
    """A 1.5 and a 1.6 um strand differ in `model_id` and in nothing a shape names."""
    other = route_for(general_guide(grid=grid, radius=1.6e-6), grid).tables
    assert other.identity["n_nodes"] == tables.identity["n_nodes"]
    with pytest.raises(OutputError, match=r"their modal\.tables\.model_id is"):
        electric_field(run, other, plane=-1, guide=guide)


def test_the_analytic_signal_comes_back_in_the_convention_asked_for(run, tables, guide):
    """`convention=`, as every other accessor of `solprop.processing` takes it.

    The reference is solved in the internal FFT convention and the rest of
    processing hands back the optics one by default; the reconstruction did not,
    which made it the one accessor whose complex output meant something else.
    """
    optics = electric_field(run, tables, plane=-1, analytic=True, guide=guide)
    internal = electric_field(
        run, tables, plane=-1, analytic=True, convention="fft", guide=guide
    )
    assert np.array_equal(optics, np.conj(internal))
    assert not np.array_equal(optics, internal)
    assert np.array_equal(
        magnetic_field(run, tables, plane=-1, analytic=True, guide=guide),
        np.conj(
            magnetic_field(
                run, tables, plane=-1, analytic=True, convention="fft", guide=guide
            )
        ),
    )
    # The real field is convention-free, and both conventions give it.
    real = electric_field(run, tables, plane=-1, guide=guide)
    assert np.array_equal(real, np.real(internal))
    assert np.array_equal(
        real, electric_field(run, tables, plane=-1, convention="fft", guide=guide)
    )


def test_a_guide_sampled_at_other_knots_is_refused_although_its_tables_agree(
    run, tables, guide
):
    """The gap a table identity cannot see, closed by the guide beside it.

    A table identity records the **run's** coarse and fine extents and the node
    count, and says nothing about the frequencies the provider was solved at. Two
    guides sampled at different interior knots therefore build tables whose
    identity agrees entry for entry, and the reconstruction would be made from
    profiles interpolated through the wrong knots.
    """
    sampling = guide.sampling
    assert sampling is not None
    moved = np.array(sampling.omega, dtype=float)
    moved[len(moved) // 2] *= 1.0 + 1e-6
    other = dataclasses.replace(
        guide, sampling=dataclasses.replace(sampling, omega=moved)
    )
    with pytest.raises(OutputError, match=r"their modal\.guide\.sampling\.omega\."):
        electric_field(run, tables, plane=-1, guide=other)
    # The tables alone do not catch it, which is why the guide is **required**:
    # the keyword has no default and no alias, so the narrower check cannot be
    # reached at all.
    with pytest.raises(TypeError, match="guide"):
        electric_field(run, tables, plane=-1)  # type: ignore[call-arg]
    # And the guide that produced the file goes through every accessor.
    assert electric_field(run, tables, plane=-1, guide=guide).shape[1] == 3
    assert reconstructed_flux(run, tables, plane=-1, guide=guide).ndim == 1
    assert instantaneous_flux(run, tables, plane=-1, guide=guide).ndim == 1
    assert magnetic_field(run, tables, plane=-1, guide=guide).shape[1] == 3
    assert coordinate_intensity(run, tables, plane=-1, guide=guide).ndim == 3


def test_a_guide_with_no_sampling_has_no_identity_to_check(run, tables, guide):
    """A guide built by `from_step_index` declares no resolution to compare.

    The translation layer's refusal, re-raised in this module's own class: an
    accessor documents one exception and a caller should not have to catch two to
    ask one question.
    """
    bare = dataclasses.replace(guide, sampling=None)
    with pytest.raises(OutputError, match="declares no sampling"):
        electric_field(run, tables, plane=-1, guide=bare)


def test_a_guide_whose_region_declares_another_response_is_refused(run, tables, grid):
    """A changed `chi3` is a different run, and the tables cannot see that either."""
    other = general_guide(grid=grid, chi3=3.0e-22)
    expected = r"their modal\.guide\.regions\.0\.kerr\.chi3"
    with pytest.raises(OutputError, match=expected):
        reconstructed_flux(run, tables, plane=-1, guide=other)


def test_an_identity_block_that_is_a_subset_is_refused_as_incomplete(tables, guide):
    """What was not compared was not checked, so a partial block is not a pass.

    `solprop.processing._examples` writes a deliberately small `tables` block: it
    exists for the label and energy doctests, which never reconstruct, and it is
    the standing exercise of this branch.
    """
    with pytest.raises(OutputError, match=r"carries no modal\.tables\.n_nodes"):
        electric_field(example_general_output(), tables, plane=-1, guide=guide)


def edited(run, entry, value):
    """Return a handler holding this run's last plane under an edited identity."""
    stored = run.metadata
    assert stored.modal is not None
    modal = dict(cast("dict[str, object]", stored.modal)) | {entry: value}
    handler = MemoryOutput()
    handler.open(dataclasses.replace(stored, modal=modal))
    handler.save(run.planes()[-1])
    return handler


def test_an_identity_block_carrying_more_than_the_reference_is_refused(
    run, tables, guide
):
    """The other side of the same check: an entry nothing here can answer for."""
    assert run.metadata.modal is not None
    fuller = dict(cast("dict[str, object]", run.metadata.modal["tables"])) | {
        "sidecar_digest": "0" * 8
    }
    with pytest.raises(OutputError, match=r"carries modal\.tables\.sidecar_digest"):
        electric_field(edited(run, "tables", fuller), tables, plane=-1, guide=guide)


def test_a_record_with_no_identity_block_at_all_is_refused(run, tables, guide):
    """A file edited by hand, or written by a version that had no such block.

    The record is **data**, so a reader may find anything under one of its keys;
    the gate says what it found rather than raising where a type was assumed.
    """
    with pytest.raises(OutputError, match=r"records no modal\.guide identity"):
        electric_field(edited(run, "guide", None), tables, plane=-1, guide=guide)
    with pytest.raises(OutputError, match=r"records no modal\.tables identity"):
        electric_field(edited(run, "tables", "a digest"), tables, plane=-1, guide=guide)


def test_a_reference_on_another_grid_is_refused(run, tables, guide):
    """The stored spectrum and the reference must live on one frequency axis.

    A guide solved for a run on a shorter time window is caught by the identity
    itself, which counts the coarse bins; what this checks is the case the
    identity cannot see, an axis of the same length and the same endpoints whose
    interior knots are somewhere else.
    """
    narrow = FieldGrid.from_limits(GENERAL_Z_END, 835e-9, (815e-9, 860e-9), 30e-15)
    assert narrow.n_omega != len(tables.omega)
    other = route_for(general_guide(grid=narrow), narrow).tables
    with pytest.raises(OutputError, match=r"their modal\.tables\.n_omega is"):
        electric_field(run, other, plane=-1, guide=guide)
    moved = np.array(tables.omega, dtype=float)
    moved[len(moved) // 2] *= 1.0 + 1e-6
    shifted = dataclasses.replace(tables, omega=moved)
    assert dict(shifted.identity) == dict(tables.identity)
    with pytest.raises(OutputError, match="rebuild the route on this run"):
        electric_field(run, shifted, plane=-1, guide=guide)


# ---------------------------------------------------------------- the whitelist


def test_the_whitelist_is_one_table_read_from_both_sides():
    """A capability is decided the same way from a problem and from a record."""
    assert declared_capability(general_modal=True, spatial=False) == "general_modal"
    assert declared_capability(general_modal=False, spatial=True) == "spatial"
    assert declared_capability(general_modal=False, spatial=False) == "modal"
    general = example_general_output().metadata
    ordinary = example_output().metadata
    assert general is not None and ordinary is not None
    assert capability_of(general) == "general_modal"
    assert capability_of(ordinary) == "modal"
    assert refusal_for("modal", ["energy", "peak_power"]) is None
    assert refusal_for("general_modal", sorted(GENERAL_MODAL_ACCESSORS)) is None


def test_the_published_accessor_list_holds_what_the_same_record_admits(run):
    """`channel_labels` in, `quadrature_error` out, `channel_amplitudes` out.

    The list is written into the schema-7 record, so a name the same record's run
    refuses must not appear in it: `quadrature_error` is refused on a general guide
    by the run loop, because a provider's node rule carries no embedded coarse rule
    to difference, and `channel_amplitudes` takes two arrays and a
    table rather than an opened handler, so no door ever asks the gate about it.
    `channel_labels` is the accessor output schema 7 exists for and is in.
    """
    assert "channel_labels" in GENERAL_MODAL_ACCESSORS
    # The reconstructed flux trace and the run's peak statistic are different
    # quantities and have different names; both are admitted.
    assert {"reconstructed_flux", "analytic_flux"} <= GENERAL_MODAL_ACCESSORS
    assert "quadrature_error" not in GENERAL_MODAL_ACCESSORS
    assert "channel_amplitudes" not in GENERAL_MODAL_ACCESSORS
    published = run.metadata.modal["accessors"]
    assert list(published) == sorted(GENERAL_MODAL_ACCESSORS)
    assert "channel_labels" in published


@pytest.mark.parametrize(
    "name",
    ["electron_density", "species_density", "pressure", "core_radius", "beam_radius"],
)
def test_a_gas_or_spatial_name_is_refused_for_its_own_reason(name):
    """Advising `analytic_flux` instead of `electron_density` would be nonsense.

    A plasma density, a gas pressure and a beam radius are unavailable on a guide
    because it has no fill and no transverse grid, not because its coordinate powers
    do not add up. The gate names that reason and offers no replacement from the
    flux family (A2 companion section 6).
    """
    message = refusal_for("general_modal", [name])
    assert message is not None
    assert message.startswith(f"{name} is unavailable for this general modal guide")
    assert "gas fill or a spatial geometry" in message
    assert "analytic_flux" not in message


def test_a_basis_independent_step_diagnostic_is_let_through_by_the_gate():
    """Section 6 admits them, so whoever owns the real reason is the one to refuse.

    `quadrature_error` is refused on this route by the run loop, for a reason about
    the node rule rather than about the basis; the gate says nothing about it.
    """
    assert refusal_for("general_modal", ["quadrature_error"]) is None
    assert refusal_for("modal", ["quadrature_error"]) is None


def test_the_legacy_processing_metrics_are_refused_on_a_general_file(run):
    """The whitelist's fourth door: `solprop.processing.energy` and `peak_power`.

    `peak_power(modes='sum')` is `peak_power_total`, which the run loop, both CPU
    backends and `solprop.processing.modal` all refuse on this basis; without this
    the same number came back through a different import.
    """
    for accessor in (legacy.energy, legacy.peak_power):
        with pytest.raises(OutputError, match="is unavailable for this nonorthogonal"):
            accessor(run, plane=-1)
    # A schema-7 run on an ordinary basis is untouched, and so is every older file:
    # the shape functionals of this module are admitted on every basis either way.
    assert float(np.ravel(legacy.energy(example_output(), plane=0))[0]) > 0.0
    assert float(np.ravel(legacy.peak_power(example_output(), plane=0))[0]) > 0.0
    assert float(np.ravel(legacy.fwhm_t(run, plane=-1))[0]) > 0.0


def test_the_legacy_gate_says_nothing_about_a_handler_with_no_record():
    """There is no basis to read, so the gate defers to the accessor's own refusal."""
    with pytest.raises(OutputError, match="carries no metadata"):
        legacy.energy(MemoryOutput(), plane=0)


def test_an_older_file_still_answers_the_legacy_metrics():
    """The committed schema-6 fixture carries no identity and is the modal run it is."""
    path = Path(__file__).resolve().parents[1] / "data" / "general-modal-schema6.h5"
    with open_file(path) as stored:
        assert stored.metadata is not None
        assert stored.metadata.schema_version == 6
        assert np.all(np.isfinite(legacy.energy(stored, plane=0)))


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_both_cpu_backends_refuse_a_general_name_on_an_ordinary_run(backend):
    """The whitelist's backend door, on a run that carries no general identity.

    A caller may build a backend and ask it for statistics directly, without the
    run loop's admission; both CPU backends check, and the message is the one the
    loop would have given.
    """
    simulation = Simulation.from_gnlse(
        0.11,
        1e-4,
        [0.0, 0.0, -1.183e-26],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=1.0,
        pulseshape="sech",
        n_save=2,
        backend=backend,
        noise=None,
    )
    problem = simulation.translate()
    built = select(backend, problem, grid=simulation.grid)
    try:
        built.set_initial(problem.initial.field0, None)
        with pytest.raises(BackendError, match="is unavailable for this modal run"):
            built.stats(StatsSpec(names=("physical_energy",)))
        assert built.stats(StatsSpec(names=("energy",)))
    finally:
        built.close()


@pytest.mark.parametrize(
    "name",
    ["energy", "energy_window", "peak_power", "peak_power_total", "peak_intensity"],
)
def test_every_refused_name_carries_its_replacement(name):
    """The message names the basis and what to use instead, and nothing else."""
    message = refusal_for("general_modal", [name])
    assert message is not None
    assert message.startswith(f"{name} is unavailable for this nonorthogonal")
    assert "analytic_flux" in message or "physical_energy" in message

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A declared basis of a split block, through the coupled flow, end to end.

A declared unitary change of basis inside an exactly degenerate block leaves
``U.H diag(beta) U`` diagonal; inside a split block it does not, and the rotated
channels propagate through ``linear.coupled_flow``. The physical case is the retained
half of the weak-guidance ``LP_11`` group -- ``TE_01`` and ``TM_01``, degenerate
under weak guidance and split by the vector correction -- whose equal-weight rotation
makes the two rotated channels beat.

The end-to-end statement is the one the degenerate case uses: **the two bases agree on
the same launch**. The rotated run propagates a coupled matrix flow and the plain run
propagates two independent diagonal channels, so the two share no execution path at all
below the guide, and their agreement is the statement that the flow is the same physics
in other coordinates.
"""

import re
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

import solprop.translate.problem as problem_module
from modal_provider_cases import (
    GENERAL_Z_END,
    SPLIT_DESCRIPTORS,
    SPLIT_LABELS,
    general_grid,
    make_provider,
    split_basis,
    split_guide,
)
from solprop.backends.numpy_ref.backend import NumpyRefBackend
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import (
    BackendError,
    DriverError,
    ModeError,
    OutputError,
    TranslationError,
)
from solprop.modes.guide import ChannelBasis, ModalGuide
from solprop.output.capability import withheld_from
from solprop.output.hdf5 import open_file
from solprop.processing.time import _refuse_by_basis
from solprop.stats import StatsSpec
from solprop.translate.general import coupled_block, general_modal_route
from solprop.translate.modal_tables import generator_is_coupled, sample_modal_table

PULSE = {
    "lambda0": 835e-9,
    "fwhm": 20e-15,
    "energy": 1e-12,
    "pulseshape": "gauss",
    "backend": "numpy_ref",
}

STEPS = 10
"""Steps every run here takes over a millimetre."""

DECLARED_WINDOW = (815e-9, 860e-9)
"""The band the guide's own linear operator is compared over, m.

Outside its declared window a canonical field grid carries the **collar**, an
absorbing continuation of the operator rather than the guide's own dispersion, and
two guides with different index tables carry different collars. Measured on this
grid, ``Re Phi`` is exactly zero at 869.3, 848.1 and 827.9 nm and reads -1.63e-1 at
808.7 nm, the first bin outside the window, rising to -4.96e5 by 1656 nm. Every
comparison between two runs below is therefore taken inside the window, where the
operator is the guide's: outside it the same comparison reads 1e-3 instead of 1e-5,
and the difference is the collar's and not the flow's.
"""

TWO_BASIS_MODULUS = 5e-5
"""Largest relative departure the two bases may show in a channel's modulus.

Dimensionless, declared before the run. The two runs share no execution path below
the guide -- one applies a Schur matrix-function action per step and the other a
diagonal exponential -- and what they do share, the index tables, are two different
interpolants of the same solves: the rotated guide tabulates the block's mean and
the plain one tabulates each channel. Measured 8.4e-6; this gate is six times it.
"""

TWO_BASIS_PHASE = 1e-4
"""Largest departure in the phase **between** the two channels, radians.

The moduli and this phase are the frame-invariant content of a two-channel state: a
common phase ramp is the co-moving frame, which the two guides choose from their own
reference channels and which differs between them by construction. Measured 2.1e-5.
"""


@pytest.fixture(scope="module")
def grid():
    """The field grid every run here propagates on."""
    return general_grid()


@pytest.fixture(scope="module")
def window(grid):
    """The coarse bins inside the declared window, where the operator is the guide's."""
    low, high = (2.0 * np.pi * C / one for one in reversed(DECLARED_WINDOW))
    return (grid.omega >= low) & (grid.omega <= high)


@pytest.fixture(scope="module")
def rotated(grid):
    """The guide in the declared rotated basis, whose generator couples."""
    return split_guide(grid)


@pytest.fixture(scope="module")
def plain(grid):
    """The same guide in the provider's own coordinates, whose generator does not."""
    return split_guide(grid, rotated=False)


def build(guide, grid, *, channel=0, length=GENERAL_Z_END, **overrides):
    """Return a simulation of one guide launching one channel."""
    return Simulation.from_waveguide(
        (),
        length,
        provider=guide,
        grid=grid,
        channel=channel,
        save_planes=[length],
        dz_max=length / STEPS,
        dz_init=length / STEPS,
        **(PULSE | overrides),
    )


@pytest.fixture(scope="module")
def rotated_run(rotated, grid):
    """The rotated guide's ten-step run, and the problem it was assembled from."""
    simulation = build(rotated, grid)
    return simulation, simulation.translate(), simulation.run()


@pytest.fixture(scope="module")
def plain_runs(plain, grid):
    """One ten-step run per channel of the guide in the provider's own coordinates."""
    return [build(plain, grid, channel=one).run() for one in (0, 1)]


# ------------------------------------------------------------------- construction


def test_the_declared_rotation_of_a_split_pair_couples_its_generator(rotated, plain):
    """The premise, measured: this pair is split and its rotation couples.

    The splitting must be far above the ``64 eps max|beta|`` floor an exactly
    degenerate block is judged at, or this is the degenerate case and not this one.
    """
    provider = make_provider(
        radius=6e-6, descriptors=SPLIT_DESCRIPTORS, indices=(1.45, 1.40)
    )
    omega = 2.0 * np.pi * C / 835e-9
    sample = provider.solve(omega, 0.0, nodes=provider.node_rule(omega, n_r=24))
    diagonal = np.diag(sample.generator)
    splitting = float(abs(diagonal[0] - diagonal[1]))
    floor = 64.0 * float(np.finfo(float).eps) * float(np.max(abs(diagonal)))
    assert splitting > 1e4 * floor
    sampling = rotated.sampling
    assert sampling is not None
    assert generator_is_coupled(_table(rotated))
    assert not generator_is_coupled(_table(plain))


def _table(guide):
    """Rebuild a guide's own dense reference, in the basis it declares."""
    sampling = guide.sampling
    assert sampling is not None
    return sample_modal_table(
        guide.field_provider,
        sampling.omega,
        np.array([sampling.z]),
        sampling.nodes,
        degenerate_blocks=sampling.degenerate_blocks,
        basis=sampling.basis,
    )


def test_the_rotated_route_carries_a_coupled_flow_and_the_plain_one_does_not(
    rotated_run, plain, grid
):
    """One ``constant_schur`` section on the rotated route; nothing on the plain one.

    And the carrier beside it is mode independent, which is what makes the product
    of the two exponentials the same equation.
    """
    _, problem, _ = rotated_run
    flow = problem.linear.coupled_flow
    tables = problem.transform.modal_tables
    assert flow is not None and tables is not None
    assert flow.section_methods == ("constant_schur",)
    assert flow.connection is None
    assert float(np.max(flow.residual_norm_bound)) == 0.0
    assert flow.certificate_id == "general-constant-schur"
    assert flow.gauge_id == tables.gauge_id
    carrier = problem.linear.phi_constant
    assert carrier is not None
    assert float(np.max(abs(carrier - carrier[:, :1]))) == 0.0
    assert build(plain, grid).translate().linear.coupled_flow is None


def test_the_coupled_block_is_the_rotation_of_the_provider_s_own_generator(
    rotated_run, grid, window
):
    """``G' - G'_00 I`` and nothing else, checked against the untransformed solve.

    The block the run propagates is the declared rotation of the guide's own
    diagonal generator less its reference channel's entry, which for an equal-weight
    rotation of a pair leaves the splitting entirely in the off-diagonal and an
    exactly zero diagonal.
    """
    _, problem, _ = rotated_run
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    block = flow.generator[0, 0]
    matrix = split_basis().transform
    provider = make_provider(
        radius=6e-6, descriptors=SPLIT_DESCRIPTORS, indices=(1.45, 1.40)
    )
    nodes = provider.node_rule(float(grid.omega[grid.active_mask][0]) * 0.99, n_r=24)
    for column in np.flatnonzero(window):
        omega = float(grid.omega[column])
        generator = provider.solve(omega, 0.0, nodes=nodes).generator
        turned = matrix.conj().T @ generator @ matrix
        expected = turned - turned[0, 0] * np.eye(2)
        # The run's block comes from the table's interpolant and the reference from
        # a fresh solve, so they agree to the guide's own declared knot accuracy and
        # not to rounding.
        assert_allclose(block[:, :, column], expected, rtol=2e-6, atol=0.0)
    assert float(np.max(abs(np.diagonal(block, axis1=0, axis2=1)))) == 0.0


# ---------------------------------------------------------------------- end to end


@pytest.mark.medium
def test_the_two_bases_agree_on_the_same_launch(
    rotated_run, plain_runs, window, record_property
):
    """The end-to-end statement, through the public entry point on ten steps.

    A pure rotated channel is the superposition ``U[:, 0]`` of the provider's own
    channels, so the plain basis reproduces the same physical launch as a linear
    combination of two of its own runs; carrying the result back through ``U.H`` is
    then the rotated run, if the coupled flow is the same physics in other
    coordinates. The two share no execution path below the guide.

    Both bounds are declared first: :data:`TWO_BASIS_MODULUS` and
    :data:`TWO_BASIS_PHASE`.
    """
    _, _, out = rotated_run
    turned = out.planes()[-1].field[:, 0, :]
    matrix = split_basis().transform
    exits = [one.planes()[-1].field[:, 0, :] for one in plain_runs]
    carried = matrix.conj().T @ (matrix[0, 0] * exits[0] + matrix[1, 0] * exits[1])
    worst = 0.0
    for row in (0, 1):
        got, want = abs(turned[row, window]), abs(carried[row, window])
        worst = max(worst, float(np.max(abs(got / want - 1.0))))
    relative = float(
        np.max(
            abs(
                np.angle(
                    (turned[1, window] / turned[0, window])
                    / (carried[1, window] / carried[0, window])
                )
            )
        )
    )
    record_property("split_block_modulus_departure", worst)
    record_property("split_block_relative_phase_departure", relative)
    assert worst < TWO_BASIS_MODULUS
    assert relative < TWO_BASIS_PHASE


@pytest.mark.medium
def test_the_flow_moves_power_between_the_rotated_channels(
    rotated_run, plain_runs, window, record_property
):
    """The physics the flow carries, and the control that carries none.

    A launch into one rotated channel arrives with power in the other, at the order
    the block's own off-diagonal predicts over the run; a launch into one of the
    provider's own channels arrives with **exactly zero** in the other, because that
    route's generator is diagonal and nothing couples it.

    The ``sin^2`` reference assumes bin-diagonal unitary evolution, and the run also
    carries the ADR-007 temporal-window absorber, a convolution in frequency that is
    not bin-diagonal: the two rows carry different spectra, so it attenuates them
    differently, and the share departs from the reference by up to 9e-3 on this
    60 fs window (5e-7 on a 600 fs one). The
    gate is 5e-2 for that reason; the two-basis comparison, where the absorber
    commutes with the constant rotation, is the tight one.
    """
    _, problem, out = rotated_run
    turned = out.planes()[-1].field[:, 0, :]
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    block = flow.generator[0, 0]
    for column in np.flatnonzero(window):
        total = abs(turned[0, column]) ** 2 + abs(turned[1, column]) ** 2
        share = float(abs(turned[1, column]) ** 2 / total)
        predicted = float(np.sin(abs(block[0, 1, column]) * GENERAL_Z_END) ** 2)
        record_property(f"split_block_share_{column}", share)
        # An order statement, not a precision one: the two agree to 6e-4 and 9e-3 of
        # themselves at the two bins of this grid's declared window, and the
        # precision statement is the two-basis comparison above.
        assert share > 0.0
        assert_allclose(share, predicted, rtol=5e-2, atol=0.0)
    for exit_field in (one.planes()[-1].field[:, 0, :] for one in plain_runs):
        rows = np.sum(abs(exit_field) ** 2, axis=1)
        assert float(np.min(rows)) == 0.0


@pytest.mark.medium
def test_the_run_records_its_flow_and_a_run_without_one_does_not(
    rotated_run, plain, grid
):
    """``modal.flow`` joins the schema-7 identity, and only where a flow exists.

    A file written without a flow carries no ``flow`` key and still resumes,
    which is why the key is written conditionally rather than as ``None`` -- the
    same rule as for ``guide.sampling.basis``.
    """
    simulation, _, _ = rotated_run
    record = simulation._modal_identity()
    assert record is not None
    flow = record["flow"]
    assert isinstance(flow, dict)
    assert flow["model"] == "declared channel basis"
    assert flow["section_methods"] == ["constant_schur"]
    assert flow["reference_channel"] == 0
    assert float(cast("float", flow["largest_coupling"])) > 0.0
    assert str(flow["gauge_id"]).endswith("+lp11(0,1)")
    without = build(plain, grid)._modal_identity()
    assert without is not None
    assert "flow" not in without


# ----------------------------------------------------------------------- admission


def test_a_declared_block_whose_exponent_is_too_large_is_refused_by_name(
    rotated, grid, monkeypatch
):
    """The block carries every channel's departure from the reference one.

    A basis declared over channels that are not a degenerate group would put their
    whole dispersion difference into the matrix exponential, which is executed by
    scaling and squaring and loses digits with the exponent's norm. The limit is
    named with the measured exponent and the frequency it was reached at.

    The physical trigger is a block of channels with nothing in common, whose
    departures are the whole dispersion; on *this* pair the exponent is 1.8e-2 over
    a ten-step schedule, so the limit is lowered here rather than the guide made
    absurd. What is under test is the door, its message and the numbers it names.
    """
    monkeypatch.setattr(problem_module, "MAX_GENERAL_FLOW_EXPONENT", 1e-3)
    with pytest.raises(TranslationError, match=re.escape("above the 0.001")):
        build(rotated, grid).translate()


def test_the_modified_noise_model_is_refused_beside_a_coupled_flow(rotated, grid):
    """ADR-008's draw is transported by a diagonal flow's unitary phase, not this.

    The input model is admitted: its draw is part of the entrance state
    (``tests/driver/test_coupled_input_noise.py``).
    """
    with pytest.raises(
        TranslationError, match="coupled flow refuses the 'modified' noise model"
    ):
        build(rotated, grid, noise="modified", seed=17).translate()
    assert build(rotated, grid, noise="input", seed=17).translate() is not None
    # The backend names it too, for a problem that reaches it by hand: the
    # translation layer refuses first, so the draw is spoofed onto a clean problem.
    simulation = build(rotated, grid)
    problem = simulation.translate()
    initial = problem.initial
    spoofed = replace(
        problem,
        initial=replace(
            initial, noise_model="modified", noise0=np.zeros_like(initial.field0)
        ),
    )
    with pytest.raises(
        BackendError, match="'modified' noise model beside a coupled flow"
    ):
        NumpyRefBackend(
            spoofed, grid=simulation.grid, modes=list(simulation.modes), profiles=None
        )


def test_zdw_is_withheld_on_a_split_block_and_admitted_on_the_plain_route(
    rotated, plain, grid, tmp_path
):
    """A coupled block's per-channel constant is its mean, so ``zdw`` is refused.

    Both rotated channels read the same mean
    ``neff`` where the plain channels differ by the splitting, so the route withholds
    ``zdw`` from its record and every door refuses it by name -- the driver's, the
    backend's, and the file's -- while the plain route still admits it.
    """
    with pytest.raises(DriverError, match="zdw is unavailable for this run"):
        build(rotated, grid, stats=StatsSpec(names=("zdw",))).run(progress="none")
    record = build(rotated, grid)._modal_identity()
    assert record is not None
    published = cast("list[str]", record["accessors"])
    assert "zdw" not in published
    assert "physical_energy" in published
    plain_record = build(plain, grid)._modal_identity()
    assert plain_record is not None
    assert "zdw" in cast("list[str]", plain_record["accessors"])
    # The plain route admits the name at the door: what it then raises is the
    # statistic's own, about this window having no zero crossing.
    with pytest.raises(ModeError, match="no zero-dispersion wavelength"):
        build(plain, grid, stats=StatsSpec(names=("zdw",))).run(progress="none")
    # The file door reads the published list and agrees with the run.
    path = tmp_path / "rotated.h5"
    build(rotated, grid).run(path, progress="none")
    handle = open_file(path)
    try:
        with pytest.raises(OutputError, match="zdw is unavailable for this run"):
            _refuse_by_basis(handle, "zdw")
        _refuse_by_basis(handle, "physical_energy")
        stored = handle.metadata
    finally:
        handle.close()
    # A general record that publishes no list withholds nothing: the list is the
    # only thing the file door reads, and every record since schema 7 carries one.
    assert stored is not None and stored.modal is not None
    assert withheld_from(stored) == ("zdw",)
    unlisted = replace(stored, modal={**stored.modal, "accessors": "none"})
    assert withheld_from(unlisted) == ()


def test_a_lockstep_batch_refuses_a_member_carrying_a_coupled_flow(rotated, grid):
    """The batch reimplements the step loop and applies neither half of the flow."""
    simulation = build(rotated, grid)
    problem = simulation.translate()
    member = NumpyRefBackend(
        problem, grid=simulation.grid, modes=list(simulation.modes), profiles=None
    )
    with pytest.raises(BackendError, match="lockstep batch does not execute"):
        NumpyBatchBackend([member])
    member.close()


def test_a_coupled_block_names_a_reference_outside_its_channels(rotated):
    """The block is measured against one of the table's own channels."""
    with pytest.raises(TranslationError, match="relative to one of the table's"):
        coupled_block(_table(rotated), general_grid(), reference=7)


def test_a_diagonal_generator_returns_no_block_at_all(plain, grid):
    """The ordinary route: nothing couples, so nothing is built."""
    assert coupled_block(_table(plain), grid) is None


@pytest.mark.medium
def test_the_flow_record_survives_a_file_and_separates_two_rotations(
    rotated, grid, tmp_path
):
    """The record is in the resume key, and it tells two blocks apart.

    A file carrying the flow must read back — the entry is JSON and a numpy scalar
    in it would not — and two different declared rotations of the same pair must
    produce two different records, so that a resume against a changed block is
    refused by the block that moved and not only by the gauge.
    """
    path = tmp_path / "split.h5"
    build(rotated, grid).run(path)
    handle = open_file(path)
    try:
        metadata = handle.metadata
        assert metadata is not None
        stored = metadata.modal
    finally:
        handle.close()
    assert stored is not None
    first = stored["flow"]
    assert isinstance(first, dict)
    assert first["section_methods"] == ["constant_schur"]
    assert float(cast("float", first["largest_coupling"])) > 0.0
    # A different rotation of the same pair: the same channels, the same tables, a
    # different block.
    angle = 0.3
    other = ChannelBasis(
        np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=complex,
        ),
        ((0, 1),),
        SPLIT_LABELS,
        "lp11-30deg(0,1)",
    )
    sampling = rotated.sampling
    assert sampling is not None
    turned = ModalGuide.from_provider(
        rotated.field_provider,
        sampling.omega,
        nodes=sampling.nodes,
        degenerate_blocks=((0, 1),),
        basis=other,
    )
    identity = build(turned, grid)._modal_identity()
    assert identity is not None
    second = identity["flow"]
    assert isinstance(second, dict)
    assert second["gauge_id"] != first["gauge_id"]
    assert second["largest_coupling"] != first["largest_coupling"]


@pytest.fixture(scope="module")
def route(rotated, grid):
    """The rotated guide's own route record, for the door tests below."""
    return general_modal_route(
        rotated, grid, z_max=GENERAL_Z_END, omega_reference=float(grid.omega0)
    )


def test_a_route_with_no_block_may_not_name_a_reference_channel(route):
    """Without a block every channel carries its own dispersion, so none is the one."""
    with pytest.raises(TranslationError, match="no coupled block names reference"):
        replace(route, coupled_generator=None, reference_channel=1)


@pytest.mark.parametrize("bad", ["shape", "nan"])
def test_a_block_that_is_not_the_route_s_own_is_refused_by_name(route, bad):
    """Square on this route's channels and coarse grid, and finite."""
    block = route.coupled_generator
    assert block is not None
    spoiled = np.zeros((3, 3, block.shape[2]), dtype=complex)
    if bad == "nan":
        spoiled = np.array(block)
        spoiled[0, 1, 0] = np.nan
    with pytest.raises(TranslationError, match="must be a finite"):
        replace(route, coupled_generator=spoiled)


def test_a_reference_channel_outside_the_block_is_refused_by_name(route):
    """The reference is one of the channels the block describes."""
    with pytest.raises(TranslationError, match="reference_channel is one of its"):
        replace(route, reference_channel=5)


def test_a_block_that_couples_nothing_is_refused_by_name(route):
    """A diagonal block is the ordinary route, which carries none at all."""
    block = route.coupled_generator
    assert block is not None
    diagonal = np.zeros_like(block)
    index = np.arange(block.shape[0])
    diagonal[index, index] = block[index, index]
    with pytest.raises(TranslationError, match="diagonal at every"):
        replace(route, coupled_generator=diagonal)

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A graded-index guide end to end: linear self-imaging and a nonlinear source.

Everything else about the general modal route is tested on a step-index strand; this
module is the one place a **radial** provider is propagated, and the one place the
route's answer is checked against an analytic closed form rather than against its own
arithmetic.

The system is the parabolic sector of ``modal_provider_cases``: a 25 um core at a
contrast of 0.005, solved as the scalar weak-guidance **LP** approximation, whose two
``l = 0`` radial channels have the ideal group spacing
``beta_g = sqrt(q^2 - 2 q sqrt(2 Delta) g)/a`` with ``g = 1`` and ``g = 3``. That
spacing sets a self-imaging length of 784.7 um, inside a millimetre.
"""

from typing import cast

import numpy as np
import pytest

from modal_provider_cases import (
    GENERAL_CHI3,
    grin_grid,
    grin_guide,
    parabolic_beta,
)
from solprop import fourier
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.backends.numpy_ref.general_modal import general_source
from solprop.constants import EPS0
from solprop.driver import Simulation
from solprop.exceptions import GeneralModalBasisWarning, TranslationError
from solprop.modes.material import MaterialKerr
from solprop.output.memory import MemoryOutput
from solprop.stepper import PIController
from solprop.translate.general import general_modal_route
from solprop.translate.modal_tables import DEFAULT_GRAM_ATOL, sample_modal_table
from solprop.translate.problem import assemble

SPARSE_LONG_END = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GeneralModalBasisWarning"
)
"""Silence the active-band basis warning on the guides this module propagates.

Every route here is built on the parabolic sector sampled over the whole active band of
``grin_grid``, which reaches 2.1 um, and the interpolated basis is 2.6e-4 from
orthonormal at 1939 nm at nine knots -- above ``gram_atol``, so warned on. The **gate**,
which is what these tests depend on, looks at the declared 1.00--1.06 um window and
measures 6.8e-6 there. The warning is asserted rather than silenced in
``test_a_sparsely_sampled_grin_guide_is_refused_between_its_knots``, so nothing silenced
here goes unchecked.
"""

GRIN_ELEMENTS = 64
"""Radial P1 elements per core radius the system is solved with.

The ideal parabolic **spacing** converges at about second order in this count:
measured 3.553e-3, 8.967e-4 and 3.993e-4 relative at 32, 64 and 96 elements at
the reference frequency. Sixty-four is the value every bound below is written
from; ninety-six would halve them and costs a solve per knot.
"""

GRIN_SPACING_ERROR = 8.967e-4
"""Relative departure of the solved group spacing from the ideal closed form.

Dimensionless, measured at :data:`GRIN_ELEMENTS`. The two propagation constants
themselves agree with the closed form to 4.5e-8 and 9.1e-7; the **difference**
is a small difference of two large numbers, so its relative error is three
orders worse than theirs and it is the difference that sets the imaging length.
"""


@pytest.fixture(scope="module")
def grid():
    """The field grid the graded-index system propagates on."""
    return grin_grid()


@pytest.fixture(scope="module")
def linear_guide(grid):
    """The parabolic LP sector with no declared response at all."""
    return grin_guide(grid, elements=GRIN_ELEMENTS, knots=9)


@pytest.fixture(scope="module")
def imaging(grid, linear_guide):
    """Return the solved spacing, the ideal one and the imaging length.

    Everything is taken at the coarse **bin** nearest the reference frequency,
    not at the reference frequency itself: that bin is what a run propagates, and
    a self-imaging length quoted at a frequency no bin carries would be a
    different length.
    """
    sampling = linear_guide.sampling
    assert sampling is not None
    table = sample_modal_table(
        linear_guide.field_provider,
        sampling.omega,
        np.array([sampling.z]),
        sampling.nodes,
    )
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    omega = float(grid.omega[index])
    solved = np.real(1j * np.diag(table.evaluate(omega, 0.0).generator))
    ideal = parabolic_beta(omega)
    return {
        "table": table,
        "index": index,
        "omega": omega,
        "solved": float(solved[0] - solved[1]),
        "ideal": float(ideal[0] - ideal[1]),
        "length": float(2.0 * np.pi / (ideal[0] - ideal[1])),
    }


def test_the_solved_group_spacing_is_the_parabolic_closed_form(
    imaging, record_property
):
    """The physics half of self-imaging, before any propagation.

    The bound is written first: at :data:`GRIN_ELEMENTS` radial elements the P1
    solution's group spacing sits :data:`GRIN_SPACING_ERROR` from the ideal
    parabolic one, and the ideal form is an **infinite** parabola while this
    guide truncates its at the core radius, so the two are not expected to agree
    beyond that.
    """
    departure = abs(imaging["solved"] / imaging["ideal"] - 1.0)
    record_property("grin_spacing_departure", departure)
    record_property("grin_imaging_length_m", imaging["length"])
    assert departure < 1.05 * GRIN_SPACING_ERROR
    # And the length that follows from it is inside a millimetre, which
    # is what makes this system propagatable at all.
    assert 7e-4 < imaging["length"] < 1e-3


@SPARSE_LONG_END
@pytest.mark.slow
def test_grin_self_imaging(grid, linear_guide, imaging, record_property):
    """Two channels launched together come back into phase at the imaging length.

    The propagation half of the self-imaging case. A superposition of the two
    ``l = 0`` radial channels beats at their spacing, and after
    ``2 pi / Delta beta`` the relative phase has turned through a full turn and
    the transverse pattern is the launched one again -- self-imaging. The run goes
    to the length the **ideal** closed form predicts, so that what is measured is
    the guide against the analytic answer and not the guide against itself.

    Three bounds, each written before the run.

    1. The residual relative phase is what the solved spacing's departure from
       the ideal one accumulates over a full turn: at most
       ``2 pi`` times :data:`GRIN_SPACING_ERROR`, 5.63e-3 rad.
    2. The run reproduces **its own guide's** spacing to the table's frequency
       interpolation, ``2 pi`` times the guide's measured ``profile_error``,
       6.3e-4 rad.
    3. An interference pattern depends on a relative phase error **quadratically**
       (``cos(2 pi + eps) = 1 - eps^2/2 + ...``), so the normalised transverse
       intensity repeats to ``eps^2 / 2``.

    **The launch is a continuous wave**, one coarse bin of both channels: the
    closed form is a statement at one frequency, a single bin of a field grid is
    exactly periodic on the window, and there is then no pulse to disperse, clip
    against the declared band or feel the temporal boundary. The run keeps the
    grid's default collar, and every statement below is invariant under it: the
    two channels carry the same continuous wave, so the collar damps them by the
    same factor (measured 0.69970238 against 0.69970344) and neither the relative
    phase nor the normalised pattern moves.
    """
    sampling = linear_guide.sampling
    assert sampling is not None
    turn = 2.0 * np.pi * GRIN_SPACING_ERROR
    interpolation = 2.0 * np.pi * float(sampling.profile_error)
    record_property("grin_phase_budget_rad", turn)
    record_property("grin_interpolation_budget_rad", interpolation)
    length, centre = imaging["length"], imaging["index"]
    simulation = Simulation.from_waveguide(
        (),
        length,
        provider=linear_guide,
        grid=grid,
        lambda0=1.03e-6,
        fwhm=40e-15,
        energy=1e-15,
        pulseshape="gauss",
        save_planes=[length],
        dz_max=length / 10,
        dz_init=length / 10,
        rtol=1e-9,
        atol=1e-16,
        backend="numpy_ref",
    )
    state = np.zeros((2, 1, grid.n_omega), dtype=complex)
    state[:, 0, centre] = 1.0
    run = cast("MemoryOutput", simulation.with_input(state).run(progress="none"))
    planes = run.planes()
    first, last = planes[0].field, planes[-1].field
    assert first is not None and last is not None
    entrance = first[:, 0, centre]
    exit_amplitudes = last[:, 0, centre]
    turned = float(
        np.angle(
            (exit_amplitudes[1] / exit_amplitudes[0]) / (entrance[1] / entrance[0])
        )
    )
    # A full turn reads as an angle near zero, which is the statement.
    record_property("grin_residual_phase_rad", turned)
    assert abs(turned) < turn
    # And that residual is the guide's own discretisation error and nothing else:
    # the run reproduces the spacing it was given to the interpolation budget.
    predicted = (imaging["solved"] - imaging["ideal"]) * length
    record_property("grin_phase_against_own_spacing_rad", turned - predicted)
    assert abs(turned - predicted) < interpolation
    # The transverse pattern is the launched one again. A pattern is defined up to
    # a global factor -- the collar's damping and the channels' common phase are
    # both one -- so the statement is the normalised intensity.
    sample = imaging["table"].evaluate(imaging["omega"], 0.0)
    before = np.einsum("qcm,m->qc", sample.electric, entrance)
    after = np.einsum("qcm,m->qc", sample.electric, exit_amplitudes)
    launched = np.abs(before) ** 2 / float(np.max(np.abs(before) ** 2))
    imaged = np.abs(after) ** 2 / float(np.max(np.abs(after) ** 2))
    residual = float(np.max(np.abs(imaged - launched)))
    record_property("grin_self_image_residual", residual)
    assert residual < turned**2 / 2.0
    # The beat is real: half a turn away the same comparison fails by order one.
    opposed = np.einsum("qcm,m->qc", sample.electric, entrance * np.array([1.0, -1.0]))
    halfway = np.abs(opposed) ** 2 / float(np.max(np.abs(opposed) ** 2))
    assert float(np.max(np.abs(halfway - launched))) > 0.9


@pytest.fixture(scope="module")
def kerr_case(grid):
    """The same sector with a Kerr core, its guide, its table and its route."""
    guide = grin_guide(
        grid,
        elements=GRIN_ELEMENTS,
        knots=9,
        kerr=MaterialKerr(chi3=GENERAL_CHI3),
    )
    sampling = guide.sampling
    assert sampling is not None
    table = sample_modal_table(
        guide.field_provider, sampling.omega, np.array([sampling.z]), sampling.nodes
    )
    route = general_modal_route(
        guide,
        grid,
        z_max=1e-3,
        omega_reference=float(grid.omega0),
        kerr_form="analytic",
    )
    return {"guide": guide, "table": table, "route": route, "nodes": sampling.nodes}


@SPARSE_LONG_END
@pytest.mark.medium
def test_a_nonlinear_step_in_the_grin_guide_is_the_direct_quadrature(
    grid, kerr_case, record_property
):
    """The route's source in a radial guide, against a separately written sum.

    The nonlinear half of item 6. The oracle is the route's own dense one: the
    analytic-signal polarisation formed at the guide's nodes from the table's
    stored profiles and projected with the reciprocal map, written out here and
    sharing not one line with the pass. It is the graded-index counterpart of
    ``tests/backends/test_general_modal.py``'s step-index quadrature.

    The bound, before the run: the two differ by floating-point summation order
    over the node rule and the transform pair, so a few epsilon times the
    reduction length -- ``64 eps sqrt(n_nodes n_to)``, 1.2e-12 relative here.
    """
    nodes, table, route = kerr_case["nodes"], kerr_case["table"], kerr_case["route"]
    index = int(np.argmin(np.abs(grid.omega - 0.98 * grid.omega0)))
    assert grid.active_mask[index]
    state = np.zeros((2, 1, grid.n_omega), dtype=complex)
    state[0, 0, index] = 1.0
    state[1, 0, index] = 0.4 - 0.2j
    problem = assemble(
        grid,
        list(kerr_case["guide"].mode_collection),
        [],
        state,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-3],
        dz_max=1e-4,
        dz_init=1e-5,
        general=route,
    )
    maps = _general_maps(problem)
    assert maps is not None
    source = (
        general_source(
            state, maps, grid.crop_index, n_to=grid.n_to, pad_scale=grid.pad_scale
        )[:, 0]
        * grid.crop_scale
    )
    omega = float(grid.omega[index])
    sample = table.evaluate(omega, 0.0)
    amplitudes = fourier.analytic_signal(fourier.inverse(state[:, 0], kind="field"))
    field = np.einsum("qcm,mt->qct", sample.electric, amplitudes)
    squares = np.einsum("qct,qct->qt", field, field)
    moduli = np.einsum("qct,qct->qt", field, field.conj()).real
    chi3 = np.where(nodes.region_index == 0, GENERAL_CHI3, 0.0)
    polarisation = (
        EPS0
        * chi3[:, None, None]
        * (2 * moduli[:, None, :] * field + squares[:, None, :] * field.conj())
        / 4
    )
    overlap = np.einsum(
        "qcn,qct,q->nt", sample.electric.conj(), polarisation, nodes.weights
    )
    rate = -1j * omega * np.linalg.solve(sample.flux_metric, overlap) / 4
    direct = fourier.forward(rate.real, kind="field")
    scale = float(np.max(np.abs(direct)))
    bound = 64.0 * np.finfo(float).eps * np.sqrt(nodes.weights.size * grid.n_to)
    departure = float(np.max(np.abs(source - direct))) / scale
    record_property("grin_source_departure", departure)
    record_property("grin_source_bound", bound)
    assert departure < bound
    # And it is not zero: a linear cladding and a zero tensor would pass the
    # comparison above trivially.
    assert scale > 0.0


def test_a_sparsely_sampled_grin_guide_is_refused_between_its_knots(grid):
    """The occupied-support gate, on a guide that really does fail it.

    Five knots over this active band leave the interpolated basis 1.615e-4 from
    orthonormal where the run's light is, while its **knot** Gram defect is
    7.6e-10 -- six orders apart, so the knot defect alone cannot tell. Adding knots
    is the fix the message names, and it works: seven give 2.02e-5 and nine give
    6.77e-6.

    The same guide shows the other side of the gate: what it measures over
    the declared window is not what the **active** band reaches. At seven knots
    the worst active bin is 6.59e-4 at 1939 nm, so the route that passes the gate
    still warns, and this is where that warning is asserted.
    """
    sparse = grin_guide(grid, elements=GRIN_ELEMENTS, knots=5)
    with pytest.raises(TranslationError, match="between its knots"):
        general_modal_route(
            sparse, grid, z_max=1e-3, omega_reference=float(grid.omega0)
        )
    with pytest.warns(GeneralModalBasisWarning, match="more knots"):
        denser = general_modal_route(
            grin_guide(grid, elements=GRIN_ELEMENTS, knots=7),
            grid,
            z_max=1e-3,
            omega_reference=float(grid.omega0),
        )
    assert denser.gram_defect < 1e-9
    assert denser.occupied_departure < 3e-5
    assert denser.active_departure > DEFAULT_GRAM_ATOL


def test_the_grin_guide_carries_its_core_response_and_nothing_on_its_exterior(grid):
    """A graded-index core may declare a response; the exterior may not.

    `GradedIndexProvider.regions` passes the declared response to the core region,
    so a graded-index guide need not be linear.
    """
    guide = grin_guide(
        grid, elements=GRIN_ELEMENTS, knots=7, kerr=MaterialKerr(chi3=1e-22)
    )
    core, exterior = guide.regions
    assert core.name == "core"
    assert core.kerr is not None and core.kerr.chi3 == 1e-22
    assert exterior.responses == ()
    # And it is in the provider's identity, so two guides differing only in what
    # their core responds with are not one guide.
    assert guide.field_provider.identity["core_responses"] == ["MaterialKerr"]
    assert (
        grin_guide(grid, elements=GRIN_ELEMENTS, knots=7).field_provider.identity[
            "core_responses"
        ]
        == []
    )

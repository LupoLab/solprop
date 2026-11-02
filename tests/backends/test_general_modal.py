# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route across the seam: its leaves, and its absence.

Kernel schema 12 adds four optional leaves and no behaviour: every route that exists
carries all four as ``None``, nothing new crosses the seam, and every backend refuses
each leaf by name until it executes that exact representation.

The refusal is not the schema check in disguise. A backend compiled against schema 12
accepts every schema-12 problem, so without these four checks a wheel would take a
problem carrying a representation it has no kernel for and silently run the old
physical model.

The second half of this module is the route itself on ``numpy_ref``: the nonlinear
source of a general modal guide against the modal Kerr coefficient and against a
separately written quadrature, its chunking, its ledger and every combination it
refuses. The translation layer's half -- the record, the
placeholders and the assembly's refusals -- is
``tests/translate/test_general_route.py``.
"""

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bounds import assert_within, fft_bound, higham_gamma
from legacy_geometry import LEGACY_GRID
from modal_provider_cases import (
    GENERAL_CHI3 as CHI3,
)
from modal_provider_cases import (
    GENERAL_Z_END as Z_END,
)
from modal_provider_cases import general_case as strand_case
from solprop import fourier
from solprop import stats as stats_module
from solprop.backends import select
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.admission import refuse_general_route
from solprop.backends.numpy_ref.backend import _GENERAL_LEDGER_ROWS, _general_maps
from solprop.backends.numpy_ref.general_modal import (
    GeneralMaps,
    delayed_polarisation,
    expand_pair_history,
    general_memory,
    general_node_chunk,
    general_source,
    node_delayed_maps,
    node_kerr_form,
    node_kerr_tensor,
    symmetric_cubic,
    symmetric_kerr_coefficients,
    symmetric_monomials,
)
from solprop.boundaries import absorber
from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.nonlinear.normalisation import modal_kerr_coefficient
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.coupled import CoupledFlowTables, ModalMetricTables, metric_at
from solprop.translate.general import metric_departure
from solprop.translate.linop import delta_phi
from solprop.translate.problem import _walk_arrays, assemble
from solprop.translate.responses import (
    RegionalResponseTables,
    isotropic_kerr_tensor,
    regional_responses,
)

LEAVES = ("coupled_flow", "modal_metric", "modal_tables", "regional_responses")
GROUP = {
    "coupled_flow": "linear",
    "modal_metric": "linear",
    "modal_tables": "transform",
    "regional_responses": "responses",
}


def leaf_value(name, problem):
    """Return a value for one leaf that the backend must refuse.

    Three of the four are built for real. ``modal_tables`` is a dense E/H reference of
    a guide this problem has no provider for, so it stands in as a sentinel installed
    past the constructor: the refusal reads the field for ``None`` and nothing else,
    and it fires before ``validate`` ever looks inside.
    """
    n_omega = problem.shape[-1]
    n_modes = problem.shape[1]
    omega = np.linspace(1.0e15, 2.0e15, n_omega)
    if name == "coupled_flow":
        return CoupledFlowTables(
            omega,
            np.array([0.0, 1.0]),
            np.zeros((1, 2, n_modes, n_modes, n_omega), dtype=complex),
            None,
            None,
            None,
            None,
            None,
            np.zeros((1, n_omega)),
            np.zeros((1, n_omega)),
            np.ones((1, n_omega)),
            ("constant_schur",),
            "lab",
            "schur",
        )
    if name == "modal_metric":
        return ModalMetricTables(
            "identity",
            omega,
            np.array([0.0]),
            None,
            None,
            np.zeros((1, n_omega)),
            np.ones((1, n_omega)),
            np.ones((1, n_omega)),
            "lab",
            "orthonormal",
        )
    if name == "regional_responses":
        return RegionalResponseTables(
            np.zeros(1, dtype=np.int32),
            np.full((1, 1, 1), 2.2e-52),
            np.ones((1, 3, 3, 3, 3)),
            np.zeros((1, 0, 3, 3, 3, 3)),
            np.zeros((1, 1, 1, 0, 16)),
            np.zeros((1, 0), dtype=np.int32),
            np.full(8, 1e-15),
            "isotropic-kerr",
            "dense-nodes",
        )
    return object()


def with_leaf(problem, name):
    """Return the problem with one general-modal leaf filled in.

    The group is copied first and the leaf set past the copy's constructor, because
    the session's runtime hook checks the annotation and the sentinel of
    ``modal_tables`` is not a ``ModalTransformTables``.
    """
    group_name = GROUP[name]
    group = dataclasses.replace(getattr(problem, group_name))
    object.__setattr__(group, name, leaf_value(name, problem))
    return dataclasses.replace(problem, **{group_name: group})


@pytest.fixture(scope="module")
def disabled(soliton_grid, soliton_problem):
    """The plain N-soliton problem: every general-modal leaf absent."""
    return soliton_problem(soliton_grid, 1, save_planes=[0.001])


def test_every_general_modal_leaf_is_absent_on_an_existing_route(disabled):
    assert disabled.linear.coupled_flow is None
    assert disabled.linear.modal_metric is None
    assert disabled.transform.modal_tables is None
    assert disabled.responses.regional_responses is None
    assert disabled.validate() is None


def test_the_disabled_route_puts_no_new_array_across_the_seam(disabled):
    """No leaf, no bytes: the walk that `validate` checks finds nothing new.

    This is the memory half of the all-`None` promise. The arithmetic half is the
    bitwise identity between the schema-11 and schema-12 sources, measured once when
    schema 12 was introduced; nothing inside one source tree can measure that.
    """
    names = [name for name, _ in _walk_arrays(disabled)]
    assert names
    for leaf in LEAVES:
        assert not [name for name in names if leaf in name], leaf


@pytest.mark.parametrize("leaf", LEAVES)
def test_every_backend_refuses_every_general_modal_leaf_by_name(
    backend_class, backend_name, disabled, soliton_grid, leaf
):
    """A leaf a backend has no kernel for is refused by its dotted name.

    `numpy_ref` executes three of the four as the general modal route and the
    fourth, `linear.coupled_flow`, as the coherent wall block, so it names all four
    in `admits` and checks the exact representation itself
    (`test_the_reference_refuses_...` below and `tests/modes/test_wall_operator.py`).
    `numba_cpu` and ``native`` execute all four too, and each runs the shared admission
    list under its own name, which refuses a spoiled leaf here and names it.
    """
    assert backend_name in BACKEND_NAMES
    if backend_name == "numpy_ref":
        pytest.skip("numpy_ref executes every schema-12 leaf in one combination")
    spoiled = with_leaf(disabled, leaf)
    expected = f"does not execute {GROUP[leaf]}.{leaf}"
    if backend_name == "native":
        expected = "native executes (a general modal guide|a constant-Schur|.* as part)"
    elif backend_name == "numba_cpu" and leaf != "coupled_flow":
        expected = (
            "numba_cpu executes (a general modal guide|.* as part of the general)"
        )
    elif backend_name == "numba_cpu":
        expected = "numba_cpu executes a constant-Schur coupled flow"
    with pytest.raises(BackendError, match=expected):
        backend_class(spoiled, grid=soliton_grid)


BACKEND_NAMES = {"numpy_ref", "numba_cpu", "native"}
"""Every backend name the refusal is exercised on, so a new one cannot slip past."""


# ---------------------------------------------- the general modal route on numpy_ref

ROUNDING = 1e-12
"""Relative bound on two differently grouped sums over the same quadrature.

Predeclared, and the same number and the same reason as the bound on the
coefficient dual: a few hundred products, summed one way by the route's transforms
and contractions and another way by the tests' direct quadrature.
"""


@pytest.fixture(scope="module")
def strand():
    """One HE11 channel of a 1.5 um step-index strand."""
    return strand_case((("HE", 1, 1, "even"),))


@pytest.fixture(scope="module")
def strand_pair():
    """The two parities of the same HE11 channel, declared degenerate."""
    return strand_case((("HE", 1, 1, "even"), ("HE", 1, 1, "odd")), blocks=((0, 1),))


def general_problem(case, state, **overrides):
    """Assemble one general modal problem on a case's guide and grid."""
    arguments = {
        "controller": PIController(rtol=1e-8, atol=1e-14),
        "save_planes": [Z_END],
        "dz_max": Z_END / 10,
        "dz_init": Z_END / 20,
        "general": case.route,
    } | overrides
    return assemble(case.grid, list(case.guide.mode_collection), [], state, **arguments)


def continuous_wave(case, amplitudes, fraction=0.98):
    """Return the bin index and a state with that bin of every channel excited.

    One bin of a field grid is a sinusoid exactly periodic on the window, so its
    analytic signal has a constant modulus and the analytic-signal Kerr
    polarisation sits at that one bin and nowhere else: no leakage can hide an
    error, and the coarse grid resolves the whole source exactly.
    """
    grid = case.grid
    index = int(np.argmin(np.abs(grid.omega - fraction * grid.omega0)))
    assert grid.active_mask[index]
    state = np.zeros((len(amplitudes), 1, grid.n_omega), dtype=complex)
    for channel, amplitude in enumerate(amplitudes):
        state[channel, 0, index] = amplitude
    return index, state


def route_source(case, state):
    """Return the route's own `da/dz` in sqrt(W)/m, prefactor included.

    The pass returns the projected polarisation times `-i omega`; the crop scale
    `n_t / n_to` is carried by `responses.prefactor`, where it is on every route, so
    the amplitude rate is the product of the two.
    """
    problem = general_problem(case, state)
    maps = _general_maps(problem)
    assert maps is not None
    source = general_source(
        state,
        maps,
        case.grid.crop_index,
        n_to=case.grid.n_to,
        pad_scale=case.grid.pad_scale,
    )
    return source[:, 0] * case.grid.crop_scale


def direct_source(case, state, omega):
    """Return `da/dz` from a separately written quadrature over the nodes.

    Not one line of the route: the channels' analytic amplitudes from
    `fourier.analytic_signal`, the nodal field of the table's own stored profiles at
    the one excited frequency, the isotropic polarisation
    `P = eps0 chi3 (2 |E|^2 E + (E.E) E*) / 4` at every node and time sample, and
    the reciprocal projection `-i omega W^-1 int(e* . P) dA / 4`, on the **coarse**
    time grid, which resolves a single-bin source exactly. Nothing here knows about
    S, Q, the pad or crop scales, the fine grid or the rank-four tensor.
    """
    sample = case.table.evaluate(omega, 0.0)
    amplitudes = fourier.analytic_signal(fourier.inverse(state[:, 0], kind="field"))
    field = np.einsum("qcm,mt->qct", sample.electric, amplitudes)
    squares = np.einsum("qct,qct->qt", field, field)
    moduli = np.einsum("qct,qct->qt", field, field.conj()).real
    chi3 = np.where(case.nodes.region_index == 0, CHI3, 0.0)
    polarisation = (
        EPS0
        * chi3[:, None, None]
        * (2 * moduli[:, None, :] * field + squares[:, None, :] * field.conj())
        / 4
    )
    overlap = np.einsum(
        "qcn,qct,q->nt", sample.electric.conj(), polarisation, case.nodes.weights
    )
    rate = -1j * omega * np.linalg.solve(sample.flux_metric, overlap) / 4
    return fourier.forward(rate.real, kind="field")


def test_one_continuous_wave_channel_is_the_modal_kerr_coefficient(
    strand, record_property
):
    """`da/dz = -i gamma |a|^2 a`, with gamma from `modal_kerr_coefficient`.

    The one statement the general source has to make before anything is propagated
    through it. Two bounds, both written before the run:

    * given the channel's own sampled flux, through `metric=`, the agreement is
      `ROUNDING`: the coefficient and the route then divide by the same `W_00`,
      and the two calculations share nothing else. This is the statement of the
      pair, and it is what the flux argument is for; it is also the metric
      argument the route's own contraction shares.
    * with unit flux assumed instead -- the default, and what a provider's
      normalisation declares over the whole physical domain -- the bound is the
      table's own **sampled Gram defect**, because the route divides by the
      sampled `W` through the `W^-1` inside Q. The two then differ by the basis's
      measured departure from orthonormal and by nothing else: that defect is what
      the table's construction gate measured and what the route's identity metric
      carries in its error row.

    Both measurements are recorded as test properties.
    """
    case = strand
    index, state = continuous_wave(case, [3.0 + 1.0j])
    omega = float(case.grid.omega[index])
    source = route_source(case, state)

    sample = case.table.evaluate(omega, 0.0)
    chi3 = np.where(case.nodes.region_index == 0, CHI3, 0.0)
    flux = float(np.real(sample.flux_metric[0, 0]))
    amplitude = fourier.analytic_signal(fourier.inverse(state[0], kind="field"))

    def rate_of(gamma):
        """Return `-i gamma |a|^2 a` on the coarse grid, as the route returns it."""
        assert gamma > 0.0
        turn = -1j * gamma * np.abs(amplitude) ** 2 * amplitude
        return fourier.forward(turn.real, kind="field")

    expected = rate_of(
        modal_kerr_coefficient(
            sample.electric[:, :, 0], case.nodes.weights, chi3, omega, metric=flux
        )
    )
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0 and case.table.gram_defect > 0.0
    record_property(
        "cw_against_gamma_with_the_sampled_flux",
        float(np.max(np.abs(source[0] - expected[0]))) / scale,
    )
    assert_allclose(source[0], expected[0], rtol=0.0, atol=ROUNDING * scale)

    unit_flux = rate_of(
        modal_kerr_coefficient(
            sample.electric[:, :, 0], case.nodes.weights, chi3, omega
        )
    )
    record_property(
        "cw_against_gamma", float(np.max(np.abs(source[0] - unit_flux[0]))) / scale
    )
    assert_allclose(
        source[0], unit_flux[0], rtol=0.0, atol=case.table.gram_defect * scale
    )


def test_the_source_of_one_channel_is_the_direct_quadrature(strand, record_property):
    """The whole chain against a hand-written quadrature, to rounding."""
    case = strand
    index, state = continuous_wave(case, [3.0 + 1.0j])
    expected = direct_source(case, state, float(case.grid.omega[index]))
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    source = route_source(case, state)
    record_property(
        "one_channel_against_the_quadrature",
        float(np.max(np.abs(source - expected))) / scale,
    )
    assert_allclose(source, expected, rtol=0.0, atol=ROUNDING * scale)


@pytest.fixture(scope="module")
def harmonic_strand():
    """The same strand on a band wide enough to keep the third harmonic.

    `general_grid`'s coarse band stops at 543 nm and the harmonic of an 835 nm
    carrier is at 278 nm, so the crop removes it there. This grid declares down to
    250 nm: its coarse top is 5 omega_0 and its active band reaches 3.64 omega_0,
    so the harmonic is **inside** the band the run keeps and the projection has to
    get it right at its own frequency. `n_to = 2 n_t` still, which is what the
    route's harmonic-room check requires.
    """
    return strand_case(
        (("HE", 1, 1, "even"),),
        # The former geometry: the knots were chosen against its frequency samples.
        grid=FieldGrid.from_limits(
            Z_END, 835e-9, (250e-9, 860e-9), 60e-15, **LEGACY_GRID
        ),
        # The band spans a factor seven in frequency where the shared case spans
        # 1.1, and the table's own held-out accuracy gate refuses nine knots there.
        knots=25,
    )


def direct_real_source(case, state, index):
    """Return `da/dz` for the **real** cubic from a separately written quadrature.

    The real-field partner of `direct_source`, and the only one that sees the third
    harmonic. The real nodal field is the real part of the table's own stored
    profiles times the channels' analytic amplitudes; the polarisation is
    `P = eps0 chi3 (E.E) E` at every node and time sample; and because that has
    content at **two** frequencies the projection is done bin by bin on the coarse
    spectrum, each bin with the reference evaluated at its own frequency and with
    that frequency's own flux metric. Nothing here knows about S, Q, the pad or
    crop scales, the fine grid or the rank-four tensor.
    """
    grid = case.grid
    amplitudes = fourier.analytic_signal(fourier.inverse(state[:, 0], kind="field"))
    profile = case.table.evaluate(float(grid.omega[index]), 0.0)
    field = np.real(np.einsum("qcm,mt->qct", profile.electric, amplitudes))
    chi3 = np.where(case.nodes.region_index == 0, CHI3, 0.0)
    squares = np.einsum("qct,qct->qt", field, field)
    polarisation = EPS0 * chi3[:, None, None] * squares[:, None, :] * field
    spectrum = fourier.forward(polarisation, kind="field")
    rate = np.zeros((state.shape[0], grid.n_omega), dtype=complex)
    for bin_index in (index, 3 * index):
        omega = float(grid.omega[bin_index])
        sample = case.table.evaluate(omega, 0.0)
        overlap = np.einsum(
            "qcn,qc,q->n",
            sample.electric.conj(),
            spectrum[..., bin_index],
            case.nodes.weights,
        )
        rate[:, bin_index] = (
            -1j * omega * np.linalg.solve(sample.flux_metric, overlap) / 4
        )
    return rate


# The harmonic fixture's band spans a factor seven in frequency, and at 25 knots
# its interpolated basis is 1.114e-4 from orthonormal at 1523.8 nm -- just over
# `gram_atol`, so the active-band warning fires. The **gate**,
# which is what this test depends on, is weighted by the launch and passes. What
# the warning is about is asserted in `tests/driver/test_general_modal.py`.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.GeneralModalBasisWarning")
@pytest.mark.medium
def test_the_real_cubic_source_is_the_direct_quadrature_with_the_harmonic(
    harmonic_strand, record_property
):
    """The real cubic against a hand-written quadrature, harmonic included.

    The oracle `direct_source` computes the **analytic-signal** polarisation, so on
    `general_grid`, whose crop removes the harmonic, it still matches the route's
    default real cubic to rounding. It says nothing about the harmonic. This
    case puts the harmonic inside the coarse band and compares the whole chain --
    the synthesis map S, the projection map Q, the node chunking and every node of
    the strand -- against a quadrature written from the real cubic
    directly, with the reference evaluated at the harmonic's **own** frequency.

    **The bound is `ROUNDING`, written before the run**: a single excited bin makes
    a cubic with content at exactly two frequencies, both inside the band and
    neither folded, so the coarse quadrature and the fine-grid route evaluate the
    same sum and nothing but its grouping separates them.
    """
    case = harmonic_strand
    index, state = continuous_wave(case, [3.0 + 1.0j])
    assert case.grid.active_mask[3 * index]
    expected = direct_real_source(case, state, index)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    # The harmonic is a real part of the answer here and not a rounding residue.
    assert abs(expected[0, 3 * index]) > 1e-3 * abs(expected[0, index])
    source = route_source(case, state)
    record_property(
        "the_real_cubic_against_the_quadrature",
        float(np.max(np.abs(source - expected))) / scale,
    )
    assert_allclose(source, expected, rtol=0.0, atol=ROUNDING * scale)


def test_two_continuous_wave_channels_carry_the_cross_terms(
    strand_pair, record_property
):
    """Two channels give the cross terms of the same contraction.

    The two parities of one HE11 channel, launched with different complex
    amplitudes, against the same separately written quadrature. The polarisation
    then carries `|a_1|^2 a_2`, `a_1 a_2 conj(a_1)` and `a_1^2 conj(a_2)` terms and
    the projection mixes the channels; a route that had only the self terms right
    passes the one-channel test above and fails this one.
    """
    case = strand_pair
    index, state = continuous_wave(case, [0.7 - 0.3j, -0.4 + 0.9j])
    omega = float(case.grid.omega[index])
    source = route_source(case, state)
    expected = direct_source(case, state, omega)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    # Both channels are driven, and neither is driven only by itself: dropping the
    # partner's amplitude changes each channel's source.
    alone = direct_source(case, state * np.array([1.0, 0.0])[:, None, None], omega)
    assert abs(alone[0, index] - expected[0, index]) > 0.01 * scale
    record_property(
        "the_pair_against_the_quadrature",
        float(np.max(np.abs(source - expected))) / scale,
    )
    assert_allclose(source, expected, rtol=0.0, atol=ROUNDING * scale)


@pytest.mark.medium
def test_a_millimetre_of_one_channel_accumulates_the_self_phase(
    strand, record_property
):
    """A millimetre of continuous wave turns by `-gamma |a|^2 z` and nothing else.

    The step loop, not the source alone: the interaction picture, the prefactor, the
    spectral collar and the controller. A single excited bin stays one bin under
    self-phase modulation, so dividing the final state by the linear flow at that
    bin leaves the nonlinear phase exactly. A millimetre keeps the propagation short;
    the controller takes about thirty-six steps over it here.

    The bound is **1e-8 relative on the accumulated phase**, written before the run:
    the controller is asked for `rtol = 1e-8` per step, and a pure phase rotation
    has no error the embedded pair cannot see.
    """
    case = strand
    length = Z_END
    index, state = continuous_wave(case, [9.5e3])
    problem = general_problem(case, state)
    backend = NumpyRefBackend(problem, grid=case.grid)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(length)
    final = backend.observe().require_field()[0, 0, 0, index]
    backend.close()
    assert summary.met_tolerance

    omega = float(case.grid.omega[index])
    sample = case.table.evaluate(omega, 0.0)
    gamma = modal_kerr_coefficient(
        sample.electric[:, :, 0],
        case.nodes.weights,
        np.where(case.nodes.region_index == 0, CHI3, 0.0),
        omega,
        metric=float(np.real(sample.flux_metric[0, 0])),
    )
    amplitude = fourier.analytic_signal(fourier.inverse(state[0], kind="field"))
    power = float(np.abs(amplitude[0, 0]) ** 2)
    linear = np.exp(delta_phi(problem.linear, 0.0, length))[0, 0, 0, index]
    turned = final / (state[0, 0, index] * linear)
    expected = -gamma * power * length
    assert abs(expected) > 1.0
    record_property(
        "a_millimetre_of_accumulated_phase",
        abs(float(np.angle(turned)) - expected) / abs(expected),
    )
    record_property("a_millimetre_of_modulus", abs(float(np.abs(turned)) - 1.0))
    assert_allclose(np.abs(turned), 1.0, rtol=1e-8, atol=0.0)
    assert_allclose(np.angle(turned), expected, rtol=1e-8, atol=0.0)


# --------------------------------------------------------------- the route's refusals


def spoiled_problem(case, *, amplitudes=(1.0,), **changes):
    """Assemble a general problem and replace one group's field past the hook."""
    _, state = continuous_wave(case, list(amplitudes))
    problem = general_problem(case, state)
    for group_name, name, value in changes.values():
        group = dataclasses.replace(getattr(problem, group_name))
        object.__setattr__(group, name, value)
        problem = dataclasses.replace(problem, **{group_name: group})
    return problem


def test_the_reference_refuses_a_diagonal_metric_by_name(strand):
    """The route is an orthonormal basis; nothing builds a diagonal metric."""
    metric = ModalMetricTables(
        "diagonal",
        strand.grid.omega,
        np.array([0.0, Z_END]),
        np.full((1, 1, 1, strand.grid.n_omega), 0.5),
        None,
        np.zeros((1, strand.grid.n_omega)),
        np.full((1, strand.grid.n_omega), 0.5),
        np.full((1, strand.grid.n_omega), 0.5),
        "lab",
        "sampled",
    )
    problem = spoiled_problem(strand, metric=("linear", "modal_metric", metric))
    with pytest.raises(BackendError, match="kind 'diagonal'"):
        NumpyRefBackend(problem, grid=strand.grid)
    absent = spoiled_problem(strand, metric=("linear", "modal_metric", None))
    with pytest.raises(BackendError, match="is absent"):
        NumpyRefBackend(absent, grid=strand.grid)


# ---------------------------------------------- the declared change of basis


@pytest.fixture(scope="module")
def circular_strand():
    """The same parity pair, declared in the circular basis, with its dense metric."""
    return strand_case(
        (("HE", 1, 1, "even"), ("HE", 1, 1, "odd")), blocks=((0, 1),), circular=(0, 1)
    )


def test_the_reference_executes_a_dense_metric_of_a_declared_basis(circular_strand):
    """The admission: a unitary transform of a certified orthonormal reference."""
    index, state = continuous_wave(circular_strand, [1.0, 0.0])
    problem = general_problem(circular_strand, state)
    assert problem.linear.modal_metric is not None
    assert problem.linear.modal_metric.kind == "dense"
    backend = NumpyRefBackend(problem, grid=circular_strand.grid)
    backend.set_initial(problem.initial.field0, None)
    values = backend.stats(StatsSpec(names=("physical_energy", "analytic_flux")))
    assert set(values) == {"physical_energy", "analytic_flux"}
    # The metric is handed over, so the numbers are the weighted ones. What sets
    # the difference is the **stored block's** departure from the identity at the
    # one bin this state occupies -- not the table's knot Gram defect, which is
    # two orders smaller -- so the bound is written from that quantity. Twice it,
    # because the spectral norm bounds the quadratic form's relative move and the
    # peak and the integral each take it once.
    block = metric_at(problem.linear.modal_metric, 0.0)
    assert block is not None
    departure = metric_departure(block[:, :, index : index + 1])
    assert departure > 0.0
    plain = {
        name: float(getattr(stats_module, name)(circular_strand.grid, state))
        for name in values
    }
    for name, value in values.items():
        assert float(value[0]) != plain[name]
        assert_allclose(float(value[0]), plain[name], rtol=2 * departure, atol=0.0)
    # The unlaunched partner carries only the whitening's rounding, so it has no
    # centroid to report: the guard is relative to the plane, not to zero.
    centroid = backend.stats(StatsSpec(names=("omega0",)))["omega0"]
    assert np.isfinite(centroid[0, 0, 0])
    assert np.isnan(centroid[0, 1, 0])
    backend.close()


def test_the_reference_refuses_a_dense_metric_of_another_basis(circular_strand):
    _, state = continuous_wave(circular_strand, [1.0, 0.0])
    problem = general_problem(circular_strand, state)
    metric = problem.linear.modal_metric
    assert metric is not None
    elsewhere = dataclasses.replace(metric, gauge_id="elsewhere")
    spoiled = spoiled_problem(
        circular_strand,
        amplitudes=(1.0, 0.0),
        metric=("linear", "modal_metric", elsewhere),
    )
    with pytest.raises(BackendError, match="describes the basis 'elsewhere'"):
        NumpyRefBackend(spoiled, grid=circular_strand.grid)


def test_the_reference_refuses_a_dense_metric_too_far_from_orthonormal(
    circular_strand,
):
    """The nonunitary refusal at the backend, measured on the block itself."""
    _, state = continuous_wave(circular_strand, [1.0, 0.0])
    problem = general_problem(circular_strand, state)
    metric = problem.linear.modal_metric
    assert metric is not None and metric.dense is not None
    skewed = dataclasses.replace(metric)
    object.__setattr__(skewed, "dense", metric.dense * 1.5)
    spoiled = spoiled_problem(
        circular_strand,
        amplitudes=(1.0, 0.0),
        metric=("linear", "modal_metric", skewed),
    )
    with pytest.raises(BackendError, match=r"dense and 0\.5"):
        NumpyRefBackend(spoiled, grid=circular_strand.grid)


def test_the_reference_refuses_an_identity_metric_that_is_not_the_identity(strand):
    """Defence in depth: a certificate that reaches a backend ungated is an error.

    The route's record gates the defect at its own declared `gram_atol`, but the
    backend executes `kind == "identity"` as the identity whatever the error row
    says, and nothing downstream reads that row again, so the backend checks it too.
    `DEFAULT_GRAM_ATOL` is the number, the same one the table and the route use.
    """
    error = np.full((1, strand.grid.n_omega), 0.5)
    metric = ModalMetricTables(
        "identity",
        strand.grid.omega,
        np.array([0.0, Z_END]),
        None,
        None,
        error,
        np.full((1, strand.grid.n_omega), 0.5),
        np.full((1, strand.grid.n_omega), 1.5),
        "lab",
        "sampled",
    )
    problem = spoiled_problem(strand, metric=("linear", "modal_metric", metric))
    with pytest.raises(BackendError, match=r"linear\.modal_metric\.error is 0\.5"):
        NumpyRefBackend(problem, grid=strand.grid)


def test_the_reference_refuses_a_metric_or_a_response_with_no_tables(
    disabled, soliton_grid
):
    """A leaf of the general route without the reference it belongs to."""
    for leaf in ("modal_metric", "regional_responses"):
        spoiled = with_leaf(disabled, leaf)
        with pytest.raises(BackendError, match=r"no transform\.modal_tables"):
            NumpyRefBackend(spoiled, grid=soliton_grid)


def test_the_reference_refuses_a_reference_with_no_execution_map(strand):
    """A table sampled over a path carries no map, and is refused."""
    tables = dataclasses.replace(strand.route.tables)
    object.__setattr__(tables, "synthesis", None)
    object.__setattr__(tables, "projection", None)
    problem = spoiled_problem(strand, maps=("transform", "modal_tables", tables))
    with pytest.raises(BackendError, match="carries neither"):
        NumpyRefBackend(problem, grid=strand.grid)


def test_the_reference_refuses_moving_maps_without_their_connection(strand):
    """Maps that move with no coupled flow are the adiabatic approximation, refused.

    ``numpy_ref`` admits a moving guide's maps only beside the
    ``"frozen_residual"`` flow that carries its connection; here the maps are spoiled
    to two coefficients along z on a problem with no flow at all.
    """
    tables = dataclasses.replace(strand.route.tables)
    assert strand.route.tables.synthesis is not None
    object.__setattr__(
        tables, "synthesis", np.repeat(strand.route.tables.synthesis, 2, axis=1)
    )
    problem = spoiled_problem(strand, maps=("transform", "modal_tables", tables))
    with pytest.raises(BackendError, match="carries no coupled flow"):
        NumpyRefBackend(problem, grid=strand.grid)
    # A backend that has not built moving maps refuses them by name, whatever else.
    with pytest.raises(BackendError, match=r"native executes .* of constant geometry"):
        refuse_general_route(problem, backend="native")


def test_the_reference_refuses_a_power_weight_that_is_not_one(strand):
    """The Poynting amplitude is the true-power amplitude: the weight is one."""
    _, state = continuous_wave(strand, [1.0])
    problem = general_problem(strand, state)
    linear = dataclasses.replace(problem.linear)
    object.__setattr__(linear, "power_weight_constant", np.full(problem.shape[1:], 1.5))
    spoiled = dataclasses.replace(problem, linear=linear)
    with pytest.raises(BackendError, match="constant one"):
        NumpyRefBackend(spoiled, grid=strand.grid)


def spatial_case():
    """Return a small radial problem and its grid, for the whitelist's spatial door."""
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = RadialGrid(6e-4, 16, collar_fraction=0.0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 80e-6
    )
    problem = assemble(
        grid,
        None,
        [],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length / 8,
        schedule=FixedSchedule.uniform(0.0, grid.length, grid.length / 8),
        stats_spec=default_spec("radial"),
    )
    return grid, problem


@pytest.mark.parametrize("name", ["numpy_ref", "numba_cpu"])
def test_both_backends_check_the_whitelist_before_the_spatial_branch(name):
    """A spatial run asked for a general name is refused, not answered.

    `numpy_ref` consulted the whitelist **after** its spatial early return, so a
    radial problem asked for `physical_energy` reached `collect_spatial` and was
    answered from a geometry that carries no certified flux metric and no stored
    E/H reference at all. Both backends now check first, which is also where the
    four-door claim of `solprop.output.capability` is true.
    """
    grid, problem = spatial_case()
    built = select(name, problem, grid=grid)
    try:
        built.set_initial(problem.initial.field0, None)
        for asked in ("physical_energy", "analytic_flux"):
            with pytest.raises(
                BackendError, match=f"{asked} is unavailable for this spatial run"
            ):
                built.stats(StatsSpec(names=(asked,)))
        # And the spatial run's own statistics still come back.
        assert built.stats(StatsSpec(names=("fwhm_r",)))
    finally:
        built.close()


@pytest.mark.parametrize("name", ["numpy_ref", "numba_cpu"])
def test_a_general_statistic_under_a_weight_that_is_not_one_is_refused(
    strand, soliton_grid, soliton_problem, name
):
    """`physical_energy` and `analytic_flux` are the unit-weight arithmetic.

    Both are outside `_POYNTING_STATISTICS`, so a weight that is not the constant
    one would reach them as the **unweighted** field's arithmetic and be silently
    wrong rather than obviously so. The combination is unreachable through the
    translation layer -- the route certifies the constant weight, and `numba_cpu`
    refuses a general guide's leaves at construction -- so each backend is handed
    the two leaves past its own admission, which is the only way to exercise a
    guard written for the day either side is loosened.
    """
    if name == "numpy_ref":
        _, state = continuous_wave(strand, [1.0])
        grid, problem = strand.grid, general_problem(strand, state)
    else:
        grid = soliton_grid
        problem = soliton_problem(soliton_grid, 1, save_planes=[0.001])
    built = select(name, problem, grid=grid)
    try:
        built.set_initial(problem.initial.field0, None)
        if name != "numpy_ref":
            object.__setattr__(problem.transform, "modal_tables", strand.route.tables)
        object.__setattr__(
            problem.linear, "power_weight_constant", np.full(problem.shape[1:], 1.5)
        )
        # Sorted, so a spec asking for both is refused by the first of the two.
        for asked, names in (
            ("analytic_flux", ("physical_energy", "analytic_flux")),
            ("physical_energy", ("physical_energy",)),
        ):
            with pytest.raises(
                BackendError, match=f"{asked} is unavailable for this run"
            ):
                built.stats(StatsSpec(names=names))
    finally:
        built.close()


def spoiled_response(strand, name, value):
    """Return the case's compiled response with one array replaced past the hook."""
    response = regional_responses(
        [CHI3], [0], n_to=strand.grid.n_to, dt_fine=strand.grid.dt_fine
    )
    object.__setattr__(response, name, value)
    return response


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("chi3", np.full((1, 2, 1), CHI3), "degree 1 along z"),
        ("delayed_tensor", np.zeros((1, 1, 3, 3, 3, 3)), "1 delayed term"),
        ("instantaneous_tensor", np.ones((1, 1, 1, 1, 1)), "1 Cartesian component"),
        ("instantaneous_tensor", np.ones((1, 2, 2, 2, 2)), "2 Cartesian component"),
        ("model_id", "isotropic-quartic", "must not be evaluated as one it does"),
    ],
)
def test_the_reference_refuses_a_regional_response_it_cannot_execute(
    strand, name, value, message
):
    """Every refusal fires before the route allocates anything.

    chi3 is constant along z, a delayed term must be the one the model id declares,
    and the tensor resolves the reference's own three components. The last is the
    dangerous one: one component **broadcasts** into the three-component contraction and
    propagates a meaningless polarisation, and two died inside `einsum` with a shape
    message naming no leaf.

    Checked through the backend's constructor rather than through
    `node_kerr_tensor`, because that is where it matters: the refusal is in the
    admission list, which runs before `StepDriver.__init__` allocates a stage.
    """
    problem = spoiled_problem(
        strand,
        responses=(
            "responses",
            "regional_responses",
            spoiled_response(strand, name, value),
        ),
    )
    with pytest.raises(BackendError, match=message):
        NumpyRefBackend(problem, grid=strand.grid)
    # And the public helper refuses the same record, before its own allocation.
    with pytest.raises(BackendError, match=message):
        node_kerr_tensor(strand.route.tables, spoiled_response(strand, name, value))


def test_an_unknown_model_identity_is_not_executed_as_one_the_backend_knows(strand):
    """A model this package does not know is refused, not read as its neighbour.

    The instantaneous form travels in `model_id` rather than in a leaf field of its
    own, because a field would move a contract row; the price is that the backend
    must refuse an identity outside the declared vocabulary instead of defaulting.
    """
    spoiled = spoiled_response(strand, "model_id", "isotropic-quartic")
    with pytest.raises(BackendError, match="isotropic-quartic"):
        node_kerr_form(spoiled)
    assert node_kerr_form(None) == "analytic"
    assert node_kerr_form(strand.route.responses) == "real"


def test_the_reference_refuses_a_state_with_a_polarisation_axis(strand):
    """A general guide's components are its own; the state keeps `n_pol = 1`.

    Two state polarisations beside a general reference would leave the pass
    returning a one-wide source for `stage_output` to broadcast onto a two-wide
    prefactor -- two polarisations driven by one channel's polarisation.
    """
    _, state = continuous_wave(strand, [1.0])
    problem = general_problem(strand, state)
    n_batch, n_modes, _, n_omega = problem.shape
    spoiled = dataclasses.replace(problem, shape=(n_batch, n_modes, 2, n_omega))
    with pytest.raises(BackendError, match="n_pol = 1, and this problem carries 2"):
        NumpyRefBackend(spoiled, grid=strand.grid)


def test_the_reference_refuses_a_state_with_a_batch_axis(strand):
    """One member at a time: the stage evaluates the polarisation of member zero.

    `_general_stage` passes `lab[0]` to the source pass, because a general guide's
    execution maps are one immutable table and the pass is written for one state.
    Without this refusal a batch of two launches would propagate one member's
    polarisation for both and nothing downstream would say so.
    """
    _, state = continuous_wave(strand, [1.0])
    problem = general_problem(strand, state)
    _, n_modes, n_pol, n_omega = problem.shape
    spoiled = dataclasses.replace(problem, shape=(2, n_modes, n_pol, n_omega))
    with pytest.raises(BackendError, match="one member at a time"):
        NumpyRefBackend(spoiled, grid=strand.grid)


def test_a_linear_guide_has_no_polarisation_at_any_node(strand):
    """A guide whose regions declare no response gives an identically zero tensor."""
    summed = node_kerr_tensor(strand.route.tables, None)
    assert summed.shape == (len(strand.nodes.weights), 3, 3, 3, 3)
    assert not np.any(summed)


def test_the_node_tensor_is_the_susceptibility_of_each_region(strand):
    """The core responds and the cladding does not, node by node."""
    summed = node_kerr_tensor(strand.route.tables, strand.route.responses)
    core = strand.route.tables.region_index == 0
    assert core.any() and not core.all()
    # The route's default cubic is the real-field one, so the record carries the
    # tensor normalised for it.
    assert strand.route.responses is not None
    assert strand.route.responses.model_id == "isotropic-kerr-real"
    expected = np.broadcast_to(CHI3 * isotropic_kerr_tensor("real"), summed[core].shape)
    assert_allclose(summed[core], expected, rtol=1e-15)
    assert not np.any(summed[~core])
    # And the analytic form's tensor is three quarters of it, node for node.
    analytic = regional_responses(
        [CHI3], [0], n_to=strand.grid.n_to, dt_fine=strand.grid.dt_fine
    )
    assert_allclose(
        node_kerr_tensor(strand.route.tables, analytic)[core],
        0.75 * summed[core],
        rtol=1e-15,
    )


def test_the_general_chunk_and_ledger_count_the_routes_own_buffers(strand):
    """The route's chunk is its own, and every row it adds is named.

    Including the record's stored reference, which this pass never reads: it is
    resident for the whole run because the problem owns it, it is the largest group
    on a guide with many channels, and a budget that did not see it would be wrong
    by that much.
    """
    _, state = continuous_wave(strand, [1.0])
    problem = general_problem(strand, state)
    backend = NumpyRefBackend(problem, grid=strand.grid)
    ledger = backend.memory_ledger()
    backend.close()
    maps = _general_maps(problem)
    tables = strand.route.tables
    assert maps is not None
    assert ledger["general_synthesis"] == maps.synthesis.nbytes
    assert ledger["general_projection"] == maps.projection.nbytes
    assert ledger["general_tensor"] == maps.tensor.nbytes
    for name in (
        "electric",
        "magnetic",
        "backward_electric",
        "backward_magnetic",
        "reciprocal_gram",
        "coordinate_dual",
    ):
        stored = getattr(tables, name)
        assert stored is not None
        assert ledger[f"general_{name}"] == stored.nbytes
    # The six delayed rows are zero on a guide that declares no delayed response,
    # and are named anyway.
    delayed_rows = [
        name
        for name in _GENERAL_LEDGER_ROWS
        if "delayed" in name
        or name
        in (
            "general_node_drive",
            "general_node_histories",
            "general_node_drive_spectrum",
        )
    ]
    assert len(delayed_rows) == 6
    assert all(ledger[name] == 0 for name in delayed_rows)
    # A constant guide evaluates no map per stage.
    assert ledger["general_stage_maps"] == 0
    assert all(
        ledger[name] > 0
        for name in _GENERAL_LEDGER_ROWS
        if name not in (*delayed_rows, "general_stage_maps")
    )
    # A record that certifies no reciprocal basis and stores no dual reports those
    # rows as zero rather than omitting them.
    bare = dataclasses.replace(tables)
    for name in (
        "backward_electric",
        "backward_magnetic",
        "reciprocal_gram",
        "coordinate_dual",
    ):
        object.__setattr__(bare, name, None)
    rows = general_memory(maps, bare, strand.grid.n_to)
    assert set(rows) == set(_GENERAL_LEDGER_ROWS)
    assert rows["general_backward_electric"] == 0
    assert rows["general_coordinate_dual"] == 0
    assert rows["general_electric"] == ledger["general_electric"] > 0
    # A smaller budget divides the chunk, and the chunk buffers alone with it.
    small = general_node_chunk(3, strand.grid.n_to, 129, 1 << 16)
    assert 1 <= small < maps.n_nodes
    with pytest.raises(BackendError, match="must be positive"):
        general_node_chunk(3, 256, 129, 0)


def test_every_general_ledger_row_is_zero_on_another_route(disabled, soliton_grid):
    """`memory_ledger` names every buffer on every route, empty or not."""
    backend = NumpyRefBackend(disabled, grid=soliton_grid)
    ledger = backend.memory_ledger()
    backend.close()
    assert all(ledger[name] == 0 for name in _GENERAL_LEDGER_ROWS)


def test_the_chunk_loop_does_not_change_the_source(strand):
    """A chunk of one node and a chunk of every node differ only in round-off."""
    _, state = continuous_wave(strand, [2.0 + 0.5j])
    problem = general_problem(strand, state)
    maps = _general_maps(problem)
    assert maps is not None
    arguments = {
        "crop_index": strand.grid.crop_index,
        "n_to": strand.grid.n_to,
        "pad_scale": strand.grid.pad_scale,
    }
    whole = general_source(state, maps, **arguments)
    single = general_source(state, dataclasses.replace(maps, node_chunk=1), **arguments)
    scale = float(np.max(np.abs(whole)))
    assert scale > 0.0 and maps.node_chunk > 1
    assert_allclose(single, whole, rtol=0.0, atol=ROUNDING * scale)


@pytest.mark.medium
def test_the_static_profile_agrees_with_the_full_one_at_matched_accuracy(
    strand, record_property
):
    """What freezing the transverse profile in frequency costs, measured.

    The static-profile against full-profile parity, on the source itself. Every route
    before this one carries **one** transverse profile for a whole run: a static real
    matrix, or an effective area and a ``gamma``. The general route carries the profile
    at every coarse bin, which is the whole point of it, so what the older model costs
    is a number this route can measure and nothing else can.

    The comparison holds everything else fixed -- the same guide, the same nodes,
    the same launch, the same cubic -- and replaces ``S`` and ``Q`` by their
    values at the bin nearest the reference frequency, which is exactly the
    static-profile model.

    The bound, before the run. The cubic carries three factors of ``S`` and one
    of ``Q``, so a profile varying by ``v`` relative across the bins the launch
    occupies moves the source by at most ``4 v`` to first order. ``v`` is measured
    from the table itself, over the **occupied** support and not the sampled band,
    for the same reason the basis gate is: outside it the stored
    profile is an explicit zero and the relative variation is one. What is
    measured is ``synthesis``'s variation, as a **proxy** for ``Q``'s: the two
    are built from the same interpolated profiles at the same bins, ``Q``
    carrying the reciprocal normalisation, so they move together across the band
    and one number serves for the ``4 v`` the bound is written from.
    """
    grid = strand.grid
    state = GaussianPulse(wavelength=835e-9, fwhm=20e-15, energy=1e-12).build(grid)
    problem = general_problem(strand, state)
    maps = _general_maps(problem)
    assert maps is not None
    arguments = {
        "crop_index": grid.crop_index,
        "n_to": grid.n_to,
        "pad_scale": grid.pad_scale,
    }
    centre = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    synthesis, projection = maps.synthesis, maps.projection
    support = np.any(np.abs(synthesis) > 0.0, axis=(0, 1, 2))
    density = np.abs(state[0, 0]) ** 2
    occupied = support & (density > 1e-6 * density.max())
    reference = synthesis[:, :, :, centre]
    variation = max(
        float(np.max(np.abs(synthesis[:, :, :, column] - reference)))
        / float(np.max(np.abs(reference)))
        for column in np.flatnonzero(occupied)
    )
    record_property("static_profile_variation", variation)
    full = general_source(state, maps, **arguments)
    frozen = general_source(
        state,
        dataclasses.replace(
            maps,
            synthesis=np.repeat(
                synthesis[:, :, :, centre : centre + 1], synthesis.shape[3], axis=3
            ),
            projection=np.repeat(
                projection[:, :, :, centre : centre + 1], projection.shape[3], axis=3
            ),
        ),
        **arguments,
    )
    scale = float(np.max(np.abs(full)))
    departure = float(np.max(np.abs((full - frozen)[:, 0][:, occupied]))) / scale
    record_property("static_profile_departure", departure)
    assert scale > 0.0
    assert 0.0 < departure < 4.0 * variation
    # It is not rounding either: freezing the profile is a real approximation, and
    # this route is the one that can say how large it is.
    assert departure > 1e3 * ROUNDING


def test_a_coarse_synthesis_is_the_fine_one_with_the_unread_columns_dropped(strand):
    """S on the coarse grid gives what S on the fine grid gave, to rounding.

    The map used to be tabulated on `omega_fine` (A2 API section 4 as accepted),
    while the padded state it multiplies is zero everywhere but `grids.crop_index`:
    every other column multiplied a zero and could not reach the source. Slice C2b's
    review moved S to the coarse grid and made the pass place the synthesised
    spectrum into the fine one instead.

    Measured on this case **before** the change, and unchanged by it: S was
    1 783 296 bytes on the fine grid and is 898 560 on the coarse one (129 bins
    against 65, a factor 1.985 at this `ffac`); the source's largest modulus is
    8.880915318652263e-4 and its value at the excited bin
    2.808392011402124e-4 - 8.425176034206358e-4j.

    The assertion here is stronger than a recorded number, and does not depend on
    this machine's transforms: the pre-change arithmetic is run beside the new pass
    over a fine map whose unread columns are filled with **junk** of unit modulus,
    and the two agree within one evaluation's rounding -- two transform round trips,
    the synthesis and the projection (:func:`bounds.fft_bound`) -- which any junk
    that was read would exceed by orders of magnitude.
    """
    case = strand
    index, state = continuous_wave(case, [3.0 + 1.0j])
    problem = general_problem(case, state)
    maps = _general_maps(problem)
    assert maps is not None
    grid = case.grid
    n_fine = fourier.n_omega_for(grid.n_to, "field")
    assert maps.synthesis.shape[-1] == grid.n_omega < n_fine

    # S as it was built before the change, with every column the coarse grid cannot
    # reach filled with something that would show up at once if it were read.
    rng = np.random.default_rng(20260919)
    fine_maps = np.exp(2j * rng.uniform(0.0, 1.0, (*maps.synthesis.shape[:3], n_fine)))
    fine_maps[..., grid.crop_index] = maps.synthesis
    expected = pre_c2b_review_source(
        state,
        fine_maps,
        maps,
        grid.crop_index,
        n_to=grid.n_to,
        pad_scale=grid.pad_scale,
    )
    source = general_source(
        state, maps, grid.crop_index, n_to=grid.n_to, pad_scale=grid.pad_scale
    )
    assert np.max(np.abs(source)) > 0.0 and np.any(state[:, 0, index])
    bound = 2 * fft_bound(grid.n_to)
    assert_within(source, expected, bound, what="coarse synthesis")


def pre_c2b_review_source(state, synthesis, maps, crop_index, *, n_to, pad_scale):
    """Evaluate the source as the pass did with S tabulated on the fine grid.

    Kept here, and nowhere else, so that the claim above is checked against
    arithmetic and not against a number pasted from a run.
    """
    n_fine = fourier.n_omega_for(n_to, "field")
    padded = np.zeros((maps.n_modes, n_fine), dtype=np.complex128)
    padded[:, crop_index] = state[:, 0, :]
    doubling = np.full(n_fine, 2.0)
    doubling[0] = 1.0
    doubling[-1] = 1.0
    padded *= pad_scale * doubling
    source = np.zeros((maps.n_modes, len(crop_index)), dtype=np.complex128)
    for first in range(0, maps.n_nodes, maps.node_chunk):
        rows = slice(first, first + maps.node_chunk)
        spectrum = np.einsum("qcmw,mw->qcw", synthesis[rows], padded)
        field = np.zeros((*spectrum.shape[:2], n_to), dtype=np.complex128)
        field[..., :n_fine] = spectrum
        field = np.fft.ifft(field, axis=-1)
        if maps.form == "real":
            # The cubic as the pass now forms it: this helper replicates
            # the synthesis before the change, and the contraction after it.
            polarisation = EPS0 * symmetric_cubic(
                symmetric_kerr_coefficients(maps.tensor[rows]), np.real(field)
            )
        else:
            pair = np.einsum("qkt,qlt->qklt", field, field.conj())
            weighted = np.einsum("qijkl,qklt->qijt", maps.tensor[rows], pair)
            polarisation = EPS0 * np.einsum("qijt,qjt->qit", weighted, field)
        projected = fourier.forward(np.real(polarisation), kind="field")[
            ..., crop_index
        ]
        source += np.einsum("mqcw,qcw->mw", maps.projection[:, rows], projected)
    return (source * maps.frequency_factor)[:, None, :]


def test_the_reference_refuses_a_general_route_on_an_envelope_grid(strand):
    """A leaf is tabulated on the canonical coarse grid, which is increasing."""
    _, state = continuous_wave(strand, [1.0])
    problem = general_problem(strand, state)
    spoiled = dataclasses.replace(problem, grid_kind="envelope")
    with pytest.raises(BackendError, match="on a field grid"):
        NumpyRefBackend(spoiled, grid=strand.grid)


@pytest.mark.medium
def test_a_linear_guide_does_not_run_the_source_pass(strand, monkeypatch):
    """A guide with no responding region skips the pass, and steps the same.

    The tensor is identically zero, so the source is zero for every state and every
    position, and the synthesis, the `3 x n_nodes` transforms, the contraction and
    the projection would spend 108 ms per step producing it (measured on this case,
    single-threaded; the short-circuit is 0.1 ms and the whole run of ten steps drops
    from 1.08 s to 0.001 s). The legacy path short-circuits the same way when nothing
    but the accumulation was compiled.

    Not asserted as a ratio, which would be a clock on a shared machine: the pass is
    replaced by something that raises, so the assertion is that it was not called.
    The state is asserted **exactly** against a backend whose short-circuit has been
    disabled, which is the only thing that could change here: the pass it skips adds
    an identically zero source, an exact no-op whatever the order of the sums.
    """
    case = strand
    plane = 1e-6
    route = dataclasses.replace(case.route, responses=None)
    assert route.responses is None
    _, state = continuous_wave(case, [9.5e3])
    problem = general_problem(
        case,
        state,
        general=route,
        save_planes=[plane],
        dz_max=plane / 4,
        dz_init=plane / 4,
    )

    def step(backend):
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(plane)
        field = backend.observe().require_field().copy()
        backend.close()
        return field

    forced = NumpyRefBackend(problem, grid=case.grid)
    assert not forced._general_responds
    forced._general_responds = True
    expected = step(forced)

    def refuse(*arguments, **keywords):
        raise AssertionError("the general source pass ran for a linear guide")

    monkeypatch.setattr("solprop.backends.numpy_ref.backend.general_source", refuse)
    short = NumpyRefBackend(problem, grid=case.grid)
    assert not short._general_responds
    field = step(short)
    assert np.any(field)
    np.testing.assert_array_equal(field.view(np.uint8), expected.view(np.uint8))


@pytest.mark.medium
def test_the_route_carries_the_in_stage_absorber(strand):
    """The in-stage collar is a term of its own beside the general source.

    It is the one part of the stage the general path shares with the fused one, so
    it is checked rather than assumed. A continuous wave fills the window, so with
    no collar its modulus is the same at the window edge as in the middle; with the
    collar it is attenuated at the edge, where the collar lives, and not in the
    middle, where the collar is one.

    Two of the three assertions are structural -- a flat modulus, and an attenuation
    of at least a per cent at the edge -- and the third is not: the middle of the
    window is compared to the uncollared run at **1 %**, because the collar's profile
    is not exactly one there and the difference is a physical leak of the boundary
    into the interior, not a rounding error. It is a bound on that leak; the measured
    value is 0.11 %, against the 12 % the edge loses.
    """
    case = strand
    plane = 1e-6
    _, state = continuous_wave(case, [9.5e3])
    traces = {}
    for scheme in (None, "nonlinear"):
        collar = (
            {}
            if scheme is None
            else {
                "absorber": absorber(case.grid),
                "absorber_scheme": scheme,
            }
        )
        problem = general_problem(
            case,
            state,
            save_planes=[plane],
            dz_max=plane / 2,
            dz_init=plane / 4,
            **collar,
        )
        assert (problem.responses.temporal_absorber_fine is not None) == (
            scheme is not None
        )
        backend = NumpyRefBackend(problem, grid=case.grid)
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(plane)
        field = backend.observe().require_field()[0, 0, 0]
        backend.close()
        traces[scheme] = np.abs(fourier.inverse(field[None], kind="field")[0])
    middle = len(traces[None]) // 2
    assert traces[None][0] == pytest.approx(traces[None][middle], rel=1e-12)
    assert traces["nonlinear"][0] < 0.99 * traces["nonlinear"][middle]
    assert traces["nonlinear"][middle] == pytest.approx(traces[None][middle], rel=0.01)


# ------------------------------------------------------------------ the real cubic


def kerr_forms(case, state):
    """Return the route's source in both cubics, from one set of tables.

    The two differ only in the tensor and in which field it contracts, so building
    them from the same `GeneralMaps` fields isolates the model from everything else
    the route does.
    """
    problem = general_problem(case, state)
    maps = _general_maps(problem)
    assert maps is not None
    assert case.route.responses is not None
    built = {}
    for form in ("analytic", "real"):
        regional = regional_responses(
            [float(case.route.responses.chi3[0, 0, 0])],
            [int(case.route.responses.response_region[0])],
            n_to=case.grid.n_to,
            dt_fine=case.grid.dt_fine,
            form=form,
        )
        built[form] = general_source(
            state,
            dataclasses.replace(
                maps,
                tensor=node_kerr_tensor(case.route.tables, regional),
                form=form,
            ),
            case.grid.crop_index,
            n_to=case.grid.n_to,
            pad_scale=case.grid.pad_scale,
        )
    return built["analytic"], built["real"]


def test_the_two_cubics_agree_on_the_fundamental_band(strand, record_property):
    """The real cubic's fundamental component is the analytic form's, exactly.

    For a real field `E = Re E_a` the cubic
    `(E.E)E` has, among its terms, exactly those with two unconjugated factors and
    one conjugated, and their sum is `Re` of the analytic-signal polarisation
    `(2 |E|^2 E_a + (E_a.E_a) E_a*)/4` — which is why the real tensor's
    normalisation is a third where the analytic one's is a quarter. On this grid
    the third harmonic of a 835 nm pulse is at 278 nm, above the coarse band's
    543 nm top, so the crop removes it and what is left in the source is the
    fundamental alone.

    **The bound is `ROUNDING`, written before the run**: the two are the same sum
    grouped differently — one over a complex field and one over its real part — so
    nothing but the reduction's rounding separates them. Measured 1.6e-16.
    """
    case = strand
    _, state = continuous_wave(case, [3.0 + 1.0j])
    analytic, real = kerr_forms(case, state)
    scale = float(np.max(np.abs(analytic)))
    assert scale > 0.0
    departure = float(np.max(np.abs(real - analytic))) / scale
    record_property("two_cubics_on_the_fundamental", departure)
    assert departure < ROUNDING
    # The third harmonic is at 278 nm and the coarse band stops at 543 nm, so the
    # crop is what removes it; that it is generated at all is the next test.
    assert 3 * case.grid.omega0 > float(case.grid.omega[-1])


def unit_maps(grid, chi3, form):
    """Return maps for one node and one channel with identity synthesis in x.

    A test instrument, not a guide: `S` and `Q` are one in the x component at every
    frequency, so the pass's arithmetic is the cubic itself and the closed form
    below can be written down.
    """
    n_omega = grid.n_omega
    synthesis = np.zeros((1, 3, 1, n_omega), dtype=complex)
    synthesis[0, 0, 0] = 1.0
    projection = np.zeros((1, 1, 3, n_omega), dtype=complex)
    projection[0, 0, 0] = 1.0
    return GeneralMaps(
        synthesis,
        projection,
        chi3 * isotropic_kerr_tensor(form)[None],
        np.ones(n_omega, dtype=complex),
        1,
        form,
    )


def test_the_real_cubic_radiates_a_third_harmonic_and_the_analytic_one_does_not(
    record_property,
):
    """The whole point of the real form, against the closed form.

    A monochromatic real field `E = Re(A e^(-i w t))` has
    `E^3 = (3|A|^2/4) Re(A e^(-i w t)) + Re((A^3/4) e^(-3 i w t))`, so the real
    cubic's polarisation carries an analytic amplitude `3 eps0 chi3 |A|^2 A / 4` at
    the fundamental and `eps0 chi3 A^3 / 4` at the third harmonic, and the
    analytic-signal form carries the first and **nothing** at the second.

    The maps here are the identity in x on one node, so the source is the
    polarisation itself and both amplitudes are closed forms. The bound is
    `ROUNDING`: two transforms and a contraction over one node.
    """
    grid = FieldGrid.from_limits(1e-3, 835e-9, (400e-9, 2e-6), 200e-15)
    # A bin whose third harmonic is another bin of the same one-sided spectrum, so
    # both live on the grid exactly and neither leaks.
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    assert 3 * index < grid.n_omega
    amplitude = 2.0 - 0.5j
    state = np.zeros((1, 1, grid.n_omega), dtype=complex)
    state[0, 0, index] = amplitude
    chi3 = 2.2e-22
    sources = {
        form: general_source(
            state,
            unit_maps(grid, chi3, form),
            grid.crop_index,
            n_to=grid.n_to,
            pad_scale=grid.pad_scale,
        )[0, 0]
        for form in ("analytic", "real")
    }
    # The pass doubles the one-sided coefficient, scales by `pad_scale = n_to/n_t`
    # and takes an inverse transform of length `n_to`, so the analytic amplitude of
    # E at that bin is `2 A / n_t`; the forward transform of the real polarisation
    # is unnormalised, so a real sinusoid of analytic amplitude Z comes back as
    # `n_to Z / 2`. Both factors are the pass's grid conventions and neither is a
    # fitted number.
    carrier = 2.0 * amplitude / grid.n_t
    gain = 0.5 * grid.n_to
    fundamental = gain * 0.75 * EPS0 * chi3 * abs(carrier) ** 2 * carrier
    harmonic = gain * 0.25 * EPS0 * chi3 * carrier**3
    assert abs(harmonic) > 0.0
    for form, expected in (("analytic", 0.0), ("real", harmonic)):
        got = sources[form]
        assert_allclose(got[index], fundamental, rtol=ROUNDING, atol=0.0)
        third = got[3 * index]
        if expected == 0.0:
            record_property(
                "analytic_third_harmonic", float(abs(third) / abs(harmonic))
            )
            assert abs(third) < ROUNDING * abs(harmonic)
        else:
            assert_allclose(third, expected, rtol=ROUNDING, atol=0.0)
            assert abs(third) > 0.0


# ------------------------------------------------------------ the delayed response


def test_the_delayed_pass_is_the_reference_tensor_response(strand):
    """Six stored histories against the reference response's own 3x3 contraction.

    The pass stores the **six** independent
    entries of the symmetric pair history and loops over responses so that each
    region's nodes meet their own kernel; the reference
    `SilicaTensorResponse.evaluate` convolves the full 3x3 drive and contracts it
    with a four-index `einsum`. The two share the kernels and the tensors and
    nothing else, so the bound is `ROUNDING`.
    """
    case = strand
    grid = case.grid
    declared = SilicaTensorResponse()
    regional = regional_responses(
        [CHI3],
        [0],
        raman=[declared],
        n_to=grid.n_to,
        dt_fine=grid.dt_fine,
        form="real",
    )
    assert regional.n_terms == 2
    tables = case.route.tables
    owner, tensors, spectra = node_delayed_maps(tables, regional)
    assert tensors is not None and spectra is not None
    core = owner == 0
    assert core.any() and not core.all()
    built = _general_maps(general_problem(case, continuous_wave(case, [1.0])[1]))
    assert built is not None
    maps = dataclasses.replace(
        built,
        node_response=owner,
        delayed_tensor=tensors,
        delayed_spectrum=spectra,
        form="real",
    )
    rng = np.random.default_rng(20260921)
    field = rng.normal(size=(owner.size, 3, grid.n_to)) * 1e8
    pair = np.einsum("qkt,qlt->qklt", field, field)
    got = delayed_polarisation(field, pair, maps, owner)
    expected = np.stack(
        [
            declared.evaluate(field[node], CHI3, kind="field", dt=grid.dt_fine)
            if owner[node] == 0
            else np.zeros((3, grid.n_to))
            for node in range(owner.size)
        ]
    )
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    assert_allclose(got, expected, rtol=0.0, atol=ROUNDING * scale)
    # A node whose region declares nothing has no delayed polarisation at all.
    assert not np.any(got[~core])


def delayed_maps(case, form, declared=SilicaTensorResponse()):  # noqa: B008 - frozen
    """Return maps with the declared delayed response installed, in one form."""
    grid = case.grid
    regional = regional_responses(
        [CHI3],
        [0],
        raman=[declared],
        n_to=grid.n_to,
        dt_fine=grid.dt_fine,
        form=form,
    )
    owner, tensors, spectra = node_delayed_maps(case.route.tables, regional)
    assert tensors is not None and spectra is not None
    built = _general_maps(general_problem(case, continuous_wave(case, [1.0])[1]))
    assert built is not None
    return owner, dataclasses.replace(
        built,
        tensor=node_kerr_tensor(case.route.tables, regional),
        node_response=owner,
        delayed_tensor=tensors,
        delayed_spectrum=spectra,
        form=form,
    )


def test_the_analytic_form_drives_the_delayed_pass_beat_free(strand):
    """`kerr_form="analytic"` beside a declared delayed response.

    A combination that once broke without a test to catch it: in
    this form `general_source` hands `delayed_polarisation` the **complex**
    analytic signal -- the drive is the beat-free `Re(E_a conj(E_a))/2` and the
    final multiplication is by the real field -- and the pass's annotation refused
    it under the session's type hook. So the first statement here is that the route
    evaluates the combination at all, and that the delayed term really contributes.

    The second is the physics, against the reference response's own `thg=False`
    branch: the two share the kernels and the tensors and nothing else, so the
    bound is `ROUNDING`. Measured 3.4e-16.
    """
    case = strand
    grid = case.grid
    declared = SilicaTensorResponse()
    owner, maps = delayed_maps(case, "analytic")
    _, state = continuous_wave(case, [3.0 + 1.0j])
    with_delayed = general_source(
        state, maps, grid.crop_index, n_to=grid.n_to, pad_scale=grid.pad_scale
    )
    without = general_source(
        state,
        dataclasses.replace(
            maps, node_response=None, delayed_tensor=None, delayed_spectrum=None
        ),
        grid.crop_index,
        n_to=grid.n_to,
        pad_scale=grid.pad_scale,
    )
    scale = float(np.max(np.abs(without)))
    assert scale > 0.0
    assert float(np.max(np.abs(with_delayed - without))) > 1e-3 * scale

    rng = np.random.default_rng(20260921)
    real_field = rng.normal(size=(owner.size, 3, grid.n_to)) * 1e8
    analytic = np.stack(
        [
            np.stack([fourier.analytic_signal(row) for row in node])
            for node in real_field
        ]
    )
    pair = np.einsum("qkt,qlt->qklt", analytic, analytic.conj())
    got = delayed_polarisation(analytic, 0.5 * np.real(pair), maps, owner)
    expected = np.stack(
        [
            declared.evaluate(
                real_field[node], CHI3, kind="field", dt=grid.dt_fine, thg=False
            )
            if owner[node] == 0
            else np.zeros((3, grid.n_to))
            for node in range(owner.size)
        ]
    )
    reference = float(np.max(np.abs(expected)))
    assert reference > 0.0
    assert_allclose(got, expected, rtol=0.0, atol=ROUNDING * reference)


@pytest.fixture(scope="module")
def raman_strand():
    """The strand with the delayed response declared, on a window long enough for it.

    `general_grid`'s window is 116 fs and this response's memory is several hundred
    -- the boson peak alone decays over 96 fs -- so the causal convolution's ramp
    would be the whole of a continuous wave there. This grid's fine window is
    1.856 ps, which leaves the ramp a small, declared fraction of the answer.
    """
    return strand_case(
        (("HE", 1, 1, "even"),),
        grid=FieldGrid.from_limits(Z_END, 835e-9, (815e-9, 860e-9), 1e-12),
        raman=SilicaTensorResponse(),
    )


@pytest.mark.medium
def test_a_continuous_wave_with_the_delayed_response_carries_the_whole_chi3(
    raman_strand, record_property
):
    """The closing statement of the two shares: they add to **one** `chi3`.

    The same statement `test_one_continuous_wave_channel_is_the_modal_kerr_
    coefficient` makes for the pure Kerr term, with the region's declared delayed
    response beside it: the instantaneous tensor keeps `1 - f_R` and the delayed
    kernels carry `(3/2) f_R`, and on the fundamental band a field-domain kernel of
    integral `X` acts like a Kerr term of `(2/3) X`, so the two add to the whole
    `chi3` and `modal_kerr_coefficient(..., chi3)` is still the coefficient.

    **The bound is declared first, and it is not `ROUNDING`.** A causal convolution
    on a finite, zero-padded window cannot reach the continuous-wave steady state
    exactly: the delayed history ramps up over the kernel's own mean delay
    `<t>_h = (w_a + w_ab) 2 tau1^2 tau2/(tau1^2 + tau2^2) = 6.4` fs (the boson
    peak's mean delay is exactly zero, its negative lobe cancelling it), so the
    window average falls short by about `f_R <t>_h / W = 8.5e-4` on this 1.856 ps
    window. The bound is **2e-3** at the excited bin and **1 %** of the source's own
    peak across the band, where the ramp's broadband spread also lands. Measured
    1.8497e-4 and 2.4221e-3.

    That is two orders inside the thing it has to separate: without the `3/2` the
    ratio would be `1 - f_R/3 = 0.9183`, off by 8.2e-2, which is forty times the
    declared bound. The exact arithmetic identity behind it is
    `test_the_two_shares_add_to_one_chi3_on_the_fundamental` in
    ``tests/nonlinear/test_silica_tensor_raman.py``.
    """
    case = raman_strand
    index, state = continuous_wave(case, [3.0 + 1.0j])
    omega = float(case.grid.omega[index])
    source = route_source(case, state)
    sample = case.table.evaluate(omega, 0.0)
    chi3 = np.where(case.nodes.region_index == 0, CHI3, 0.0)
    flux = float(np.real(sample.flux_metric[0, 0]))
    amplitude = fourier.analytic_signal(fourier.inverse(state[0], kind="field"))
    gamma = modal_kerr_coefficient(
        sample.electric[:, :, 0], case.nodes.weights, chi3, omega, metric=flux
    )
    expected = fourier.forward(
        (-1j * gamma * np.abs(amplitude) ** 2 * amplitude).real, kind="field"
    )
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    at_bin = float(abs(source[0, index] - expected[0, index]) / abs(expected[0, index]))
    record_property("cw_with_the_delayed_response_at_the_bin", at_bin)
    record_property(
        "cw_with_the_delayed_response",
        float(np.max(np.abs(source[0] - expected[0]))) / scale,
    )
    assert at_bin < 2e-3
    assert_allclose(source[0], expected[0], rtol=0.0, atol=1e-2 * scale)
    # And it is the whole chi3 and not 1 - f_R/3 of it, by a wide margin.
    assert at_bin < 0.1 * SilicaTensorResponse().fraction / 3.0


def test_the_six_stored_entries_rebuild_the_symmetric_history(strand):
    """`expand_pair_history` is the inverse of taking the upper triangle."""
    rng = np.random.default_rng(7)
    stored = rng.normal(size=(4, 6, 8))
    full = expand_pair_history(stored)
    assert full.shape == (4, 3, 3, 8)
    assert_allclose(full, np.swapaxes(full, 1, 2), rtol=0.0, atol=0.0)
    for entry, (row, column) in enumerate(
        ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    ):
        assert_allclose(full[:, row, column], stored[:, entry], rtol=0.0, atol=0.0)


UNIT = 2.0**-53
"""Unit roundoff of float64."""


def gamma(count):
    """Return Higham's ``gamma_n = n u / (1 - n u)``."""
    return count * UNIT / (1.0 - count * UNIT)


@pytest.mark.parametrize("components", [1, 2, 3])
def test_the_monomials_are_the_sorted_triples_in_lexicographic_order(components):
    """One per multiset of three components: 1, 4 and 10 of them, sorted, in order."""
    monomials = symmetric_monomials(components)
    assert len(monomials) == components * (components + 1) * (components + 2) // 6
    assert list(monomials) == sorted(monomials)
    assert all(j <= k <= last for j, k, last in monomials)
    assert len(set(monomials)) == len(monomials)


def test_the_symmetric_coefficients_sum_each_monomials_orderings_in_order():
    """The sequential sum over the orderings, lexicographic, to its rounding.

    Checked against an independent loop that visits the orderings in that order: a
    sum of at most six terms, so each side is within ``gamma_5 sum |t|`` of exact
    arithmetic, whatever order an implementation adds them in. The isotropic real
    tensor's coefficients are its closed form, ``(E.E) E_i``: one for ``x x x`` in
    ``x``, one for each ``x y y`` and ``x z z``, nothing else -- sums of ones and
    zeros, exact.
    """
    rng = np.random.default_rng(11)
    tensor = rng.standard_normal((5, 3, 3, 3, 3))
    coefficients = symmetric_kerr_coefficients(tensor)
    assert coefficients.shape == (5, 3, 10)
    for position, monomial in enumerate(symmetric_monomials(3)):
        orderings = [
            ordering
            for ordering in np.ndindex(3, 3, 3)
            if tuple(sorted(ordering)) == monomial
        ]
        expected = np.zeros((5, 3))
        magnitude = np.zeros((5, 3))
        for ordering in orderings:
            expected = expected + tensor[:, :, *ordering]
            magnitude = magnitude + np.abs(tensor[:, :, *ordering])
        limit = 2 * higham_gamma(max(1, len(orderings) - 1)) * magnitude
        assert np.all(np.abs(coefficients[:, :, position] - expected) <= limit)
    isotropic = symmetric_kerr_coefficients(isotropic_kerr_tensor("real")[None])[0]
    names = ["".join("xyz"[c] for c in m) for m in symmetric_monomials(3)]
    assert dict(zip(names, isotropic[0], strict=True)) == {
        "xxx": 1.0,
        "xxy": 0.0,
        "xxz": 0.0,
        "xyy": 1.0,
        "xyz": 0.0,
        "xzz": 1.0,
        "yyy": 0.0,
        "yyz": 0.0,
        "yzz": 0.0,
        "zzz": 0.0,
    }


@pytest.mark.parametrize("components", [1, 2, 3])
def test_the_symmetric_cubic_is_the_dense_contraction_to_rounding(components):
    """``sum_m C_im (E_j E_k) E_l`` is ``sum_jkl A_ijkl E_j E_k E_l`` for any tensor.

    Bound, stated first, per element: the symmetric order is within ``gamma_17 M_i``
    of exact arithmetic and the dense one within ``gamma_14 M_i`` (`symmetric_cubic`),
    with ``M_i = sum_jkl |A_ijkl| |E_j| |E_k| |E_l|``, so the two are within
    ``gamma_31 M_i`` of each other. A random tensor with no symmetry at all, and
    random signs, so that cancellation is exercised.
    """
    rng = np.random.default_rng(23 + components)
    shape = (4, *(components,) * 4)
    tensor = rng.standard_normal(shape)
    field = rng.standard_normal((4, components, 64))
    symmetric = symmetric_cubic(symmetric_kerr_coefficients(tensor), field)
    pair = np.einsum("qkt,qlt->qklt", field, field)
    dense = np.einsum(
        "qijt,qjt->qit", np.einsum("qijkl,qklt->qijt", tensor, pair), field
    )
    majorant = np.einsum(
        "qijkl,qjt,qkt,qlt->qit",
        np.abs(tensor),
        np.abs(field),
        np.abs(field),
        np.abs(field),
    )
    departure = np.abs(symmetric - dense)
    assert np.all(departure <= gamma(31) * majorant)
    assert np.max(departure / majorant) > 0.0


def test_the_real_form_runs_the_symmetric_contraction(strand):
    """The reference's real-form polarisation is `symmetric_cubic`'s, to rounding.

    The pass written out by hand with the symmetric contraction, on a band-limited
    random launch over the strand's every node: the same operations, so within one
    evaluation's two transform round trips (:func:`bounds.fft_bound`).
    """
    rng = np.random.default_rng(5)
    grid = strand.grid
    state = (
        grid.active_mask * np.exp(1j * rng.uniform(0.0, 2 * np.pi, grid.n_omega)) * 1e3
    )[None, None, :]
    maps = _general_maps(general_problem(strand, state))
    assert maps is not None and maps.form == "real"
    source = general_source(
        state, maps, grid.crop_index, n_to=grid.n_to, pad_scale=grid.pad_scale
    )
    doubling = np.full(fourier.n_omega_for(grid.n_to, "field"), 2.0)
    doubling[0] = doubling[-1] = 1.0
    scaled = state[:, 0, :] * (grid.pad_scale * doubling[grid.crop_index])
    rebuilt = np.zeros_like(source[:, 0])
    for first in range(0, maps.n_nodes, maps.node_chunk):
        rows = slice(first, first + maps.node_chunk)
        spectrum = np.einsum("qcmw,mw->qcw", maps.synthesis[rows], scaled)
        buffer = np.zeros((*spectrum.shape[:2], grid.n_to), dtype=complex)
        buffer[..., grid.crop_index] = spectrum
        field = np.real(np.fft.ifft(buffer, axis=-1))
        polarisation = EPS0 * symmetric_cubic(
            symmetric_kerr_coefficients(maps.tensor[rows]), field
        )
        projected = fourier.forward(polarisation, kind="field")[..., grid.crop_index]
        rebuilt += np.einsum("mqcw,qcw->mw", maps.projection[:, rows], projected)
    assert np.any(rebuilt)
    bound = 2 * fft_bound(grid.n_to)
    assert_within(source[:, 0], rebuilt * maps.frequency_factor, bound)


def test_a_delayed_record_whose_identity_disagrees_with_it_is_refused(strand):
    """The model identity and the term axis must say the same thing."""
    grid = strand.grid
    regional = regional_responses(
        [CHI3],
        [0],
        raman=[SilicaTensorResponse()],
        n_to=grid.n_to,
        dt_fine=grid.dt_fine,
    )
    object.__setattr__(regional, "model_id", "isotropic-kerr")
    with pytest.raises(BackendError, match="must agree"):
        node_delayed_maps(strand.route.tables, regional)
    plain = regional_responses([CHI3], [0], n_to=grid.n_to, dt_fine=grid.dt_fine)
    object.__setattr__(plain, "model_id", "isotropic-kerr+silica-tensor-raman")
    with pytest.raises(BackendError, match="must agree"):
        node_delayed_maps(strand.route.tables, plain)


def test_a_delayed_response_refuses_a_non_uniform_temporal_weight(strand):
    """A transform product is a convolution only against a uniform weight."""
    grid = strand.grid
    regional = regional_responses(
        [CHI3],
        [0],
        raman=[SilicaTensorResponse()],
        n_to=grid.n_to,
        dt_fine=grid.dt_fine,
    )
    weight = np.array(regional.temporal_weight, dtype=float)
    weight[3] *= 1.5
    object.__setattr__(regional, "temporal_weight", weight)
    with pytest.raises(BackendError, match="not uniform"):
        node_delayed_maps(strand.route.tables, regional)


def test_a_record_with_more_delayed_terms_than_the_model_has_is_refused(strand):
    """Two channels, named; a third is not executed as one of them."""
    grid = strand.grid
    regional = regional_responses(
        [CHI3],
        [0],
        raman=[SilicaTensorResponse()],
        n_to=grid.n_to,
        dt_fine=grid.dt_fine,
    )
    object.__setattr__(regional, "delayed_tensor", np.zeros((1, 3, 3, 3, 3, 3)))
    object.__setattr__(
        regional, "delayed_kernel", np.zeros((1, 1, 1, 3, 2 * grid.n_to))
    )
    object.__setattr__(regional, "kernel_valid_samples", np.zeros((1, 3), np.int32))
    with pytest.raises(BackendError, match="delayed channels"):
        node_delayed_maps(strand.route.tables, regional)


def test_two_regions_meet_their_own_kernels_and_only_theirs(strand):
    """Material separation: one region declares the delayed model and one does not.

    The "mixture/material separation" check. A convolution is not linear in
    its kernel's owner, so the pass loops over **responses** and each region's nodes
    meet that region's kernel; a region that declares only the instantaneous cubic
    contributes nothing delayed at all, and its nodes are not driven by its
    neighbour's history.
    """
    case = strand
    grid = case.grid
    declared = SilicaTensorResponse()
    regional = regional_responses(
        [CHI3, 0.5 * CHI3],
        [0, 1],
        raman=[declared, None],
        n_to=grid.n_to,
        dt_fine=grid.dt_fine,
        form="real",
    )
    assert regional.n_terms == 2
    # The region with no delayed declaration carries a zero tensor and zero kernels
    # on the shared term axis, which is how one record holds both.
    assert not np.any(regional.delayed_tensor[1])
    assert not np.any(regional.delayed_kernel[0, 0, 1])
    assert np.any(regional.delayed_tensor[0]) and np.any(regional.delayed_kernel)
    # And the instantaneous share is 1 - f_R only where the delayed term is.
    assert regional.instantaneous_tensor[0, 0, 0, 0, 0] == pytest.approx(
        (1.0 - declared.fraction) * isotropic_kerr_tensor("real")[0, 0, 0, 0]
    )
    assert regional.instantaneous_tensor[1, 0, 0, 0, 0] == pytest.approx(
        isotropic_kerr_tensor("real")[0, 0, 0, 0]
    )
    tables = case.route.tables
    owner, tensors, spectra = node_delayed_maps(tables, regional)
    assert tensors is not None and spectra is not None
    core, clad = owner == 0, owner == 1
    assert core.any() and clad.any() and not np.any(owner < 0)
    built = _general_maps(general_problem(case, continuous_wave(case, [1.0])[1]))
    assert built is not None
    maps = dataclasses.replace(
        built,
        node_response=owner,
        delayed_tensor=tensors,
        delayed_spectrum=spectra,
        form="real",
    )
    rng = np.random.default_rng(20260922)
    field = rng.normal(size=(owner.size, 3, grid.n_to)) * 1e8
    pair = np.einsum("qkt,qlt->qklt", field, field)
    got = delayed_polarisation(field, pair, maps, owner)
    assert not np.any(got[clad])
    expected = np.stack(
        [
            declared.evaluate(field[node], CHI3, kind="field", dt=grid.dt_fine)
            for node in np.flatnonzero(core)
        ]
    )
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    assert_allclose(got[core], expected, rtol=0.0, atol=ROUNDING * scale)
    # The ledger's six delayed rows are non-zero once a delayed term exists.
    rows = general_memory(maps, tables, grid.n_to)
    delayed_rows = (
        "general_node_drive",
        "general_node_histories",
        "general_node_drive_spectrum",
        "general_node_delayed",
        "general_delayed_tensor",
        "general_delayed_spectrum",
    )
    assert all(rows[name] > 0 for name in delayed_rows)


# ---------------------------------------------------------- the route on ``native``


@pytest.mark.parametrize(
    "raman", [None, SilicaTensorResponse()], ids=["kerr", "tensor_raman"]
)
def test_native_runs_ten_steps_of_the_route_as_numpy_ref_does(
    backend_name, backend_class, raman
):
    """Ten fixed steps of the parity pair on the native backend, against the reference.

    The bound is written before the run, as solprop-native derives it
    (``tests/test_general_modal.py`` there): per step, sixteen epsilon of the field
    for the shared linear work, and on each stage's nonlinear increment -- at most
    the window's departure from the linear run over the step count, ``rho`` -- the
    source's own rounding: two transforms at the approved ``512 eps log2 n`` each
    (two more on the doubled axis for the delayed pair), and the channel, tensor and
    node sums at ``n eps``. Ten steps add, each amplifying by ``1 + 3 rho``.
    """
    if backend_name != "native":
        pytest.skip("the Kokkos route's parity; numpy_ref is its oracle")
    case = strand_case(
        (("HE", 1, 1, "even"), ("HE", 1, 1, "odd")), blocks=((0, 1),), raman=raman
    )
    grid = case.grid
    pulse = GaussianPulse(wavelength=835e-9, fwhm=20e-15, energy=2e-9).build(grid)
    pulse = np.asarray(pulse).reshape(-1)
    state = np.stack([pulse, 0.6 * np.exp(0.3j) * pulse])[:, None, :]
    steps, length = 10, 2e-4
    problem = general_problem(
        case,
        state,
        save_planes=[length],
        schedule=FixedSchedule(np.full(steps, length / steps)),
    )

    def run(implementation, one):
        backend = implementation(one, grid=grid)
        try:
            backend.set_initial(one.initial.field0, None)
            backend.propagate_to(length)
            return backend.observe().require_field()
        finally:
            backend.close()

    expected = run(NumpyRefBackend, problem)
    actual = run(backend_class, problem)
    linear = run(
        NumpyRefBackend,
        dataclasses.replace(
            problem,
            responses=dataclasses.replace(problem.responses, regional_responses=None),
        ),
    )
    eps = float(np.finfo(float).eps)
    n_to = problem.grids.n_to
    tables = problem.transform.modal_tables
    assert tables is not None
    nodes = int(tables.region_index.size)
    source = 2 * 512 * eps * np.log2(n_to) + (2 + 84 + 3 * nodes) * eps
    if raman is not None:
        source += 2 * 512 * eps * np.log2(2 * n_to) + 27 * eps
    scale = float(np.max(np.abs(problem.initial.field0)))
    rho = float(np.max(np.abs(expected - linear))) / scale / steps
    stages = problem.stepper.tableau.n_stages
    bound = steps * (16 * eps + stages * source * rho) * (1 + 3 * rho) ** steps * scale
    assert rho > 1e-3
    assert float(np.max(np.abs(actual - expected))) <= bound

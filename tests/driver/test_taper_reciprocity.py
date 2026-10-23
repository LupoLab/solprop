# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A tapered wall run backwards transfers as the forward one's transpose.

The mirrored lossless transfer. What the implemented flow is, per channel, at one bin:
the kernel state is the vacuum amplitude ``a``; the true-power amplitude is
``b = W^(1/2) a`` with ``W = diag(Re n_eff)`` (ADR-030,
:func:`solprop.translate.linop.power_weight_at`), and

``a' = [phi I + G - W^(-1/2) C W^(1/2) + N] a``,

with ``phi`` the scalar carrier (the lowest channel's propagation constant, the frame
and ``-w_1'/(2 w_1)``), ``G`` the condensed wall block with its diagonal carrier
shift, **complex symmetric in the field amplitude** (which is ``a`` up to one
channel-independent scale), ``C`` the Bessel connection, real and antisymmetric in
``b``, and ``N = -diag(w_n'/w_n - w_1'/w_1)/2`` the per-channel
normalisation. In ``b`` the diagonal ``N`` and the scalar's ``w_1`` term cancel
exactly, and

``b' = [psi I + W^(1/2) G W^(-1/2) - C] b``.

Mirroring the radius, ``a_m(s) = a(L - s)``, keeps what depends on the local geometry
(``psi``, ``G``, ``W``) and flips what is proportional to ``a'`` (``C``). If the even
part were symmetric and the odd part antisymmetric, the mirrored generator would be
the forward one's transpose at ``L - s`` and the mirrored transfer the forward one's
transpose. ``C`` qualifies in ``b``; ``G`` does not: ``G_b = W^(1/2) G W^(-1/2)`` has
the antisymmetric part ``Delta_mn = G_mn (sqrt(w_m/w_n) - sqrt(w_n/w_m))``. No
diagonal gauge repairs both, since ``G`` is symmetric only in ``a`` (up to a scalar)
and ``C`` antisymmetric only in ``b``: the implemented model is **not exactly
reciprocal** in any amplitude, and the defect is set by ``Delta``, first order in the
weights' spread. So the statement tested is, in ``b``,

``T_b,m = T_b^T + D``, ``D = int_0^L U_m(L, s) Delta(L - s) U^T(L, L - s) ds``

(Duhamel), which in the kernel's amplitude reads
``T_a,m = W(0)^(-1) T_a^T W(L) + W(0)^(-1/2) D W(L)^(1/2)``. The runs are linear --
one bin at ``1e-9`` sqrt(W) in one channel at a time, no temporal collar, at the
carrier, which is inside the scored band -- over a fifth of a millimetre
in ten and in five steps.
"""

from dataclasses import replace

import numpy as np
import pytest

import solprop.translate.problem as problem_module
from driver.test_coherent_capillary import LENGTH, linear_launch, varying
from driver.test_flow_collar import backend_class
from solprop.exceptions import GrazingIncidenceWarning, MaterialRangeWarning
from solprop.modes import CoherentWall, wall_operator
from solprop.modes.capillary import WallImpedance
from solprop.profiles import PiecewiseLinear
from solprop.stepper import FixedSchedule
from solprop.translate.linop import delta_phi, power_weight_at

pytestmark = [
    pytest.mark.filterwarnings(
        f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
    ),
    pytest.mark.filterwarnings(
        f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
    ),
    # The helper builds the weak collar before it is removed, and it says so.
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]
"""The qualification case's deliberate warnings (``test_coherent_capillary``)."""

WIDE, NARROW = 50e-6, 40e-6
"""The taper's two radii, m: the qualification taper."""

CORNERS = (0.0, 4e-5, 1.6e-4, LENGTH)
"""Straight, transition, straight, m: symmetric about ``L/2``, so the mirrored taper
has the same corners, sections and Chebyshev nodes, and both schedules land on them."""


def transfer(radii, steps, name="numpy_ref"):
    """Return one taper's true-power transfer at the carrier bin, and its pieces.

    Column ``j`` is the end state of a run launched in channel ``j`` alone, divided by
    the launched amplitude, and taken to ``b`` at both ends.
    """
    simulation = varying(
        wall=CoherentWall(),
        radius=PiecewiseLinear.from_stages(CORNERS, list(radii)),
        dz_max=LENGTH / steps,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule(np.full(steps, LENGTH / steps)),
    )
    simulation.settings = replace(simulation.settings, absorber=None)
    problem, grid = simulation.translate(), simulation.grid
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.connection is not None
    backend = backend_class(name)
    index = linear_launch(problem, grid)[0]
    columns = []
    for channel in range(flow.n_modes):
        _, launched = linear_launch(problem, grid, channel=channel)
        driver = backend(launched, grid=grid)
        driver.set_initial(launched.initial.field0, None)
        driver.propagate_to(LENGTH)
        columns.append(driver.observe().require_field()[0, :, 0, index] / 1e-9)
        driver.close()
    carrier = delta_phi(problem.linear, 0.0, LENGTH)[0, :, 0, index]
    assert np.all(carrier == carrier[0])
    root = {z: np.sqrt(power_weight_at(problem, z)[:, 0, index]) for z in (0, LENGTH)}
    vacuum = np.stack(columns, axis=1)
    true_power = root[LENGTH][:, None] * vacuum / root[0][None, :]
    return true_power, simulation, index


def placement_defect(simulation, index, z):
    """Return ``||Delta||`` and ``||W^(1/2) G W^(-1/2) - G||`` at `z`, in 1/m.

    ``Delta`` is the antisymmetric part the weights give the field-amplitude block in
    ``b``, doubled; the second norm bounds how far the true-power block's Hermitian
    part departs from the field block's, which is negative semidefinite (the passivity
    :func:`solprop.modes.wall_operator` checks).
    """
    grid = simulation.grid
    block = wall_operator(simulation.modes, grid.omega, z=z, active=grid.active_mask)
    field = block.generator[:, :, index]
    bin_omega = grid.omega[index : index + 1]
    weight = np.array(
        [float(np.real(mode.neff(bin_omega, z))[0]) for mode in simulation.modes]
    )
    true_power = np.sqrt(weight[:, None] / weight[None, :]) * field
    return (
        float(np.linalg.norm(true_power - true_power.T, 2)),
        float(np.linalg.norm(true_power - field, 2)),
    )


@pytest.mark.medium
@pytest.mark.parametrize("name", ["numpy_ref", "numba_cpu", "native"])
def test_a_mirrored_taper_transfers_as_the_transpose_of_the_forward_one(
    name, record_property
):
    """``T_b,m = T_b^T`` to the model's own placement defect and the discretisation.

    Bound, stated first. (i) Each ten-step transfer is within its discretisation
    error ``e_10`` of the model's, and the scheme converges at an order ``p >= 1``
    here (measured 4.87 and 4.95), so ``||T_10 - T_5|| >= e_5 - e_10 >=
    e_10``: the refinement differences of the two tapers majorise their errors.
    (ii) The model's defect ``D`` is bounded by Duhamel, ``||D|| <= K^2
    int ||Delta||``, with ``K = exp(int ||E||)``, ``E = W^(1/2) G W^(-1/2) - G``,
    bounding each partial propagator: in ``b`` the connection is skew and the field
    block with its carrier is passive, so the logarithmic norm is at most ``||E||``.
    ``||Delta||`` and ``||E||`` scale as ``1/a^4`` through the block and the weights'
    spread, monotone in the radius, so each straight takes its own radius and the
    transition the narrower one; the monotonicity is checked at samples. (iii) The
    Kerr coupling of a 1e-9 sqrt(W) launch, below 1e-12, and the rounding of the four
    transfers: the parity bound, 5.5e-12 of the launch per ten-step run on the
    compiled backends, taken as 1e-11 each. The statement is not empty: the
    forward transfer is itself **not** symmetric, and its conversion out of each
    channel is resolved, both by more than ten times the bound.
    """
    forward, mirrored = {}, {}
    for steps in (5, 10):
        forward[steps], simulation, index = transfer(
            (WIDE, WIDE, NARROW, NARROW), steps, name
        )
        mirrored[steps], _, _ = transfer((NARROW, NARROW, WIDE, WIDE), steps, name)
    samples = np.linspace(CORNERS[1], CORNERS[2], 9)
    norms = np.array([placement_defect(simulation, index, z) for z in samples])
    assert np.all(np.diff(norms, axis=0) >= 0.0)
    lengths = np.diff(CORNERS)
    delta_integral = float(lengths @ norms[[0, -1, -1], 0])
    gain = float(np.exp(2.0 * lengths @ norms[[0, -1, -1], 1]))
    bound = (
        np.linalg.norm(forward[10] - forward[5], 2)
        + np.linalg.norm(mirrored[10] - mirrored[5], 2)
        + gain * delta_integral
        + 1e-12
        + 4e-11
    )
    measured = np.linalg.norm(mirrored[10] - forward[10].T, 2)
    asymmetry = np.linalg.norm(forward[10] - forward[10].T, 2)
    conversion = np.max(abs(forward[10] - np.diag(np.diag(forward[10]))))
    for label, value in (
        ("bound", bound),
        ("placement", gain * delta_integral),
        ("measured", measured),
        ("asymmetry", asymmetry),
        ("conversion", conversion),
    ):
        record_property(label, float(value))
    assert measured <= bound
    assert asymmetry > 10 * bound
    assert conversion > 10 * bound


@pytest.mark.medium
def test_with_the_wall_block_placed_in_true_power_the_flow_is_reciprocal(
    monkeypatch, record_property
):
    """The scheme itself is reciprocal: the defect above is the placement, not the flow.

    A counterfactual, to isolate the discretisation. The wall block is placed
    symmetric in ``b`` instead of in ``a`` -- ``G_a = W^(-1/2) G W^(1/2)``, entry by
    entry ``sqrt(w_n/w_m) G_mn`` off the diagonal -- which makes the model exactly
    reciprocal in ``b`` (``Delta = 0``, the module docstring), so whatever is left is
    the frozen-residual scheme's. Bound, stated first: the two refinement
    differences, the Kerr coupling (1e-12) and the rounding of four ``numpy_ref``
    runs (1e-12), exactly as in the test above with its placement term removed.
    """
    original = problem_module._wall_generator_at

    def placed_in_true_power(
        block, grid, z, sampled, impedance: WallImpedance = "mean"
    ):
        generator = original(block, grid, z, sampled, impedance)
        band = grid.omega[grid.active_mask]
        weight = np.stack([np.real(np.asarray(mode.neff(band, z))) for mode in block])
        ratio = np.ones(generator.shape)
        ratio[:, :, grid.active_mask] = np.sqrt(weight[None] / weight[:, None])
        return generator * ratio

    monkeypatch.setattr(problem_module, "_wall_generator_at", placed_in_true_power)
    forward = {
        steps: transfer((WIDE, WIDE, NARROW, NARROW), steps)[0] for steps in (5, 10)
    }
    mirrored = {
        steps: transfer((NARROW, NARROW, WIDE, WIDE), steps)[0] for steps in (5, 10)
    }
    bound = (
        np.linalg.norm(forward[10] - forward[5], 2)
        + np.linalg.norm(mirrored[10] - mirrored[5], 2)
        + 2e-12
    )
    measured = np.linalg.norm(mirrored[10] - forward[10].T, 2)
    record_property("bound", float(bound))
    record_property("measured", float(measured))
    assert measured <= bound
    assert np.linalg.norm(forward[10] - forward[10].T, 2) > 10 * bound

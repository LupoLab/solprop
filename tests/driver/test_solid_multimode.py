# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The solid nonlinear ladder: few-mode four-wave mixing, selection rules, GRIN.

Every test here propagates the general modal route and holds it against something the
route does not compute:

* ``test_vector_fwm``: intermodal four-wave mixing in a few-mode step-index guide
  against an **independent few-mode overlap ODE** -- coupled-amplitude equations on a
  handful of discrete frequencies, with the overlaps summed directly from the
  provider's own fields at those frequencies and integrated with SciPy's DOP853;
* ``test_stepindex_fwm_selection``: the channels the launch's symmetry forbids stay
  at the rounding floor while the allowed ones grow;
* ``test_grin_mode_exchange_converges``: nonlinear mode exchange in a graded-index
  guide, refined independently in radial elements, z step and time window.

**The few-mode system.** A 6 um silica-like core in a 1.40 cladding, keeping the
``HE11`` parity pair and the four members of the ``LP11`` group, ``TE01``,
``TM01`` and the ``HE21`` pair. The core is large because the guide must stay bound
over the grid's whole active band, which reaches twice the longest declared
wavelength; the retained set is a subset of what it guides, which is a truncation
the ODE shares (it propagates the same six channels), so it is not an error of
either.

**Continuous waves, not pulses.** One coarse bin of a field grid is a sinusoid
exactly periodic on the window: a pump bin and a signal bin one bin apart generate
the idler exactly on the next bin and further products only on the same comb, so
the route's answer is a handful of complex numbers the ODE can be written for.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from modal_provider_cases import (
    GENERAL_CHI3,
    GRIN_AXIS_INDEX,
    GRIN_CONTRAST,
    GRIN_RADIUS,
    make_provider,
)
from solprop.backends import select
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import EPS0
from solprop.grid import FieldGrid
from solprop.modes import GradedIndexProvider, PowerLawPermittivity
from solprop.modes.guide import ModalGuide
from solprop.modes.material import IndexMaterial, MaterialKerr
from solprop.nonlinear.normalisation import modal_kerr_coefficient
from solprop.pulses import GaussianPulse
from solprop.stats import energy
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.general import general_modal_route
from solprop.translate.modal_tables import sample_modal_table
from solprop.translate.problem import assemble

FWM_LENGTH = 1e-4
"""Length of every few-mode run, m: a tenth of a millimetre in ten fixed steps.

The intermodal process below is **not** phase matched: ``2 beta_HE11(omega_p) -
beta_TE01(omega_s) - beta_TE01(omega_i)`` is about 2.0e4 1/m, a coherence length of
157 um. Ten steps of 10 um resolve that oscillation at 0.2 rad per step, and the
length keeps the idler within one coherence length, where it is largest.
"""

FWM_STEPS = 10
"""Fixed steps per few-mode run: the qualification scope's ten."""

FWM_DESCRIPTORS = (
    ("HE", 1, 1, "even"),
    ("HE", 1, 1, "odd"),
    ("TE", 0, 1, "even"),
    ("TM", 0, 1, "even"),
    ("HE", 2, 1, "even"),
    ("HE", 2, 1, "odd"),
)
"""The six retained channels, in state order: the HE11 pair and the LP11 group."""

HE11_EVEN, HE11_ODD, TE01, TM01, HE21_EVEN, HE21_ODD = range(6)

SYMMETRY_CLASS = {
    HE11_EVEN: (+1, -1),
    HE11_ODD: (-1, +1),
    TE01: (-1, -1),
    TM01: (+1, +1),
    HE21_EVEN: (+1, +1),
    HE21_ODD: (-1, -1),
}
"""Each channel's characters under the two mirrors ``y -> -y`` and ``x -> -x``.

A transverse vector field transforms under ``y -> -y`` as
``(E_x, E_y)(x, y) -> (E_x, -E_y)(x, -y)`` and under ``x -> -x`` as
``(-E_x, E_y)(-x, y)``. The channels' leading transverse forms -- ``x`` for HE11
even, ``y`` for odd, ``phi`` for TE01, ``r`` for TM01, ``(x, -y)/r`` and
``(y, x)/r`` for the HE21 pair -- give these characters, and the longitudinal
components follow the same ones. Written down from the fields' forms and not
measured: the test measures what they predict.
"""

PUMP_POWER = 1.4e6
"""Continuous-wave pump power, W, in HE11 even.

Half a radian of self-phase over :data:`FWM_LENGTH` (0.519 rad measured), so that
the idler is a sizeable fraction of the signal without cascading far along the comb.
A test configuration, not a physical one: a continuous wave of this power would
damage silica.
"""

SIGNAL_POWER = 1.4e2
"""Continuous-wave signal power, W: a ten-thousandth of the pump's, small signal."""


@pytest.fixture(scope="module")
def few_mode():
    """The few-mode step-index guide, its grid, table and analytic-form route."""
    grid = FieldGrid.from_limits(FWM_LENGTH, 835e-9, (800e-9, 875e-9), 150e-15)
    provider = make_provider(
        radius=6e-6,
        descriptors=FWM_DESCRIPTORS,
        kerr=MaterialKerr(chi3=GENERAL_CHI3),
        indices=(1.45, 1.40),
    )
    active = grid.omega[grid.active_mask]
    low, high = float(active[0]), float(active[-1])
    nodes = provider.node_rule(0.99 * low, n_r=24)
    guide = ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * low, 1.01 * high, 13),
        nodes=nodes,
        degenerate_blocks=((HE11_EVEN, HE11_ODD), (HE21_EVEN, HE21_ODD)),
    )
    sampling = guide.sampling
    assert sampling is not None
    table = sample_modal_table(
        provider,
        sampling.omega,
        np.array([sampling.z]),
        sampling.nodes,
        degenerate_blocks=sampling.degenerate_blocks,
    )
    # The analytic-signal cubic: the fundamental band alone, which is what the
    # ODE's discrete-frequency sums carry. The real cubic would add a third
    # harmonic the ODE does not have and the route would filter.
    route = general_modal_route(
        guide,
        grid,
        z_max=FWM_LENGTH,
        omega_reference=float(grid.omega0),
        kerr_form="analytic",
    )
    pump = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    return SimpleNamespace(
        grid=grid,
        provider=provider,
        nodes=nodes,
        guide=guide,
        table=table,
        route=route,
        pump=pump,
    )


def to_state(grid, power):
    """Return the stored spectral value of a one-bin continuous wave of `power` W.

    A single stored bin ``s`` of a field grid is the real field
    ``(2/n_t) Re(s exp(i omega t))``, whose analytic signal has the constant
    modulus ``2 |s| / n_t``; that modulus squared is the channel's power.
    """
    return np.sqrt(power) * grid.n_t / 2.0


def cw_launch(case, channels):
    """Return a state with the pump and the signal in the named channels."""
    grid, pump = case.grid, case.pump
    state = np.zeros((len(FWM_DESCRIPTORS), 1, grid.n_omega), dtype=complex)
    pump_channel, signal_channel = channels
    state[pump_channel, 0, pump] = to_state(grid, PUMP_POWER)
    state[signal_channel, 0, pump - 1] = to_state(grid, SIGNAL_POWER)
    return state


def equal_steps(length, steps):
    """Return a schedule of `steps` equal steps over `length`, landing on its end."""
    return FixedSchedule.uniform(0.0, length, length / steps)


def run_fixed(case, state, steps, backend="numpy_ref"):
    """Propagate `state` over :data:`FWM_LENGTH` in `steps` equal fixed steps."""
    problem = assemble(
        case.grid,
        list(case.guide.mode_collection),
        [],
        state,
        # A replayed schedule does not consult the controller, but the backend logs
        # a step whose embedded error exceeds it; 1e-6 is above what ten steps of
        # this case make, so a log line would mean something.
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[FWM_LENGTH],
        dz_max=FWM_LENGTH,
        dz_init=FWM_LENGTH / steps,
        general=case.route,
        schedule=equal_steps(FWM_LENGTH, steps),
    )
    owner = select(backend, problem, grid=case.grid)
    try:
        owner.set_initial(problem.initial.field0, None)
        owner.propagate_to(FWM_LENGTH)
        return np.array(owner.observe().require_field()[0, :, 0, :])
    finally:
        owner.close()


def rounding_bound(case, state):
    """Return the rounding floor a symmetry-forbidden channel is held to.

    A forbidden channel's source is a node sum whose mirror-image terms cancel
    **exactly** in exact arithmetic: the angular rule integrates every quartic
    product of these channels exactly (``StepIndexProvider.node_rule``), so what is
    left is the rounding of that sum and of the transforms around it. The same
    budget, for the same reason, as the route's source against its direct
    quadrature (``tests/driver/test_grin_system.py``), ``64 eps sqrt(n_nodes
    n_to)``, taken relative to the **launched** amplitude: the source is at most
    the launched amplitude times the accumulated nonlinear phase, which is below
    one here (:func:`pump_phase`, asserted), so over the run the forbidden
    amplitude cannot collect more than one such rounding of the launched amplitude.
    """
    eps = float(np.finfo(float).eps)
    n_nodes = int(case.nodes.weights.size)
    return 64.0 * eps * np.sqrt(n_nodes * case.grid.n_to) * float(np.max(np.abs(state)))


def pump_phase(case):
    """Return the pump's self-phase ``gamma P L`` over the run, radians.

    ``gamma`` is the channel's modal Kerr coefficient from the table at the pump
    bin (:func:`solprop.nonlinear.normalisation.modal_kerr_coefficient`), with its
    own sampled flux.
    """
    omega = float(case.grid.omega[case.pump])
    sample = case.table.evaluate(omega, 0.0)
    gamma = modal_kerr_coefficient(
        sample.electric[:, :, HE11_EVEN],
        case.nodes.weights,
        np.where(case.nodes.region_index == 0, GENERAL_CHI3, 0.0),
        omega,
        metric=float(np.real(sample.flux_metric[HE11_EVEN, HE11_EVEN])),
    )
    return gamma * PUMP_POWER * FWM_LENGTH


@pytest.mark.slow
@pytest.mark.parametrize(
    ("signal_channel", "allowed"),
    [(HE11_EVEN, (HE11_EVEN,)), (TE01, (HE11_EVEN, TE01, HE21_ODD))],
    ids=["he11_signal", "te01_signal"],
)
def test_stepindex_fwm_selection(
    few_mode, test_backend_name, signal_channel, allowed, record_property
):
    """A channel the launch's symmetry forbids stays at rounding; an allowed one grows.

    The isotropic Kerr polarisation of fields in the mirror classes ``c_1, c_2,
    ...`` is a sum of cubic products, and each product is in the class
    ``c_a c_b c_c`` of its three factors -- for these one-dimensional classes,
    whose characters are signs, a product of signs. Its projection on a channel
    of any other class vanishes by the mirror symmetry of the guide, so that
    channel's source is **zero**, not small.

    * An HE11-even pump and an HE11-even signal: every product is in HE11 even's
      class, and all five other channels are forbidden -- an LP01-like launch
      cannot feed the LP11 sector, nor the orthogonal polarisation.
    * An HE11-even pump and a TE01 signal: the products are in the pump's class and
      TE01's, so HE11 odd, TM01 and HE21 even are forbidden and HE21 odd, in
      TE01's class, is allowed -- the pump rotates TE01 into it.

    The bound, written before the run, is :func:`rounding_bound` on the forbidden
    channels, over every bin. The allowed channels of the idler's class -- the
    signal's -- must carry the idler at more than a thousandth of the signal's
    amplitude: the small-signal idler is of order ``gamma P L`` times the
    signal, 0.5 here, reduced by the phase mismatch over :data:`FWM_LENGTH` by
    at most a factor of a few. Run on the backend the lane selects, so that the
    rounding claim is made for each compiled route and not only for the reference.
    """
    case = few_mode
    phase = pump_phase(case)
    record_property("selection_pump_phase_rad", phase)
    assert 0.1 < phase < 1.0
    state = cw_launch(case, (HE11_EVEN, signal_channel))
    final = run_fixed(case, state, FWM_STEPS, backend=test_backend_name)
    pump_class = SYMMETRY_CLASS[HE11_EVEN]
    signal_class = SYMMETRY_CLASS[signal_channel]
    classes = {pump_class, signal_class}
    predicted = tuple(
        channel
        for channel in range(len(FWM_DESCRIPTORS))
        if SYMMETRY_CLASS[channel] in classes
    )
    # The class algebra the docstring states, checked before it is used.
    assert predicted == allowed
    forbidden = [c for c in range(len(FWM_DESCRIPTORS)) if c not in allowed]
    bound = rounding_bound(case, state)
    worst = float(np.max(np.abs(final[forbidden])))
    record_property("selection_forbidden_worst", worst / float(np.max(np.abs(state))))
    record_property("selection_bound", bound / float(np.max(np.abs(state))))
    assert worst < bound
    # And the allowed side is not zero: the idler, one bin above the pump, is in
    # the class of pump times pump times signal, which is the signal's, and grows in
    # every channel of that class. (The pump's own class, where it differs, is fed
    # at the pump's and the signal's bins but has no idler.)
    idler = case.pump + 1
    signal = float(np.abs(state[signal_channel, 0, case.pump - 1]))
    for channel in (c for c in allowed if SYMMETRY_CLASS[c] == signal_class):
        grown = float(np.abs(final[channel, idler]))
        record_property(f"selection_idler[{FWM_DESCRIPTORS[channel]}]", grown / signal)
        assert grown > 1e-3 * signal


def overlap_ode(case, bins, length):
    """Integrate the coupled-amplitude equations of the comb `bins` over `length`.

    The independent reference. The channels' fields at each bin are the
    **provider's own**, solved at that bin's frequency on the guide's nodes -- no
    table, no interpolation. In the interaction picture ``C_mj = c_mj exp(i beta_mj
    z)``, with the analytic amplitudes ``c_mj`` of channel ``m`` at bin ``j``, the
    field at the nodes is ``E_j = sum_m e_mj c_mj`` and the analytic-signal cubic at
    bin ``o`` is the sum over every ordered triple of comb bins that lands on it,

    ``P_o = eps0 chi3 / 4 [2 sum_{j - k + l = o} (E_j . E_k*) E_l
    + sum_{j + k - l = o} (E_j . E_k) E_l*]``,

    written out term by term in frequency and not in time: there is no transform,
    no fine grid, no stepper and no linear operator in it. The amplitude rate is
    the reciprocal projection ``-i omega_o W_o^-1 int e_o* . P_o dA / 4``. The
    equations are integrated with DOP853 at a relative tolerance of 1e-12, and
    the running integral of each rate's modulus is carried beside them: it is what
    a phase or profile error in the route multiplies (``test_vector_fwm``).

    Returns
    -------
    tuple
        The final interaction-picture amplitudes ``(n_bins, n_modes)``, the
        integrated rate moduli of the same shape, and the propagation constants.
    """
    grid, nodes = case.grid, case.nodes
    samples = [
        case.provider.solve(float(grid.omega[b]), 0.0, nodes=nodes) for b in bins
    ]
    electric = np.array([sample.electric for sample in samples])
    beta = np.array([-np.imag(np.diag(sample.generator)) for sample in samples])
    inverse_flux = np.array([np.linalg.inv(sample.flux_metric) for sample in samples])
    omega = grid.omega[bins]
    chi3 = np.where(nodes.region_index == 0, GENERAL_CHI3, 0.0)
    n_bins, n_modes = beta.shape
    launch = np.zeros((n_bins, n_modes), dtype=complex)
    centre = int(np.flatnonzero(bins == case.pump)[0])
    launch[centre, HE11_EVEN] = np.sqrt(PUMP_POWER)
    launch[centre - 1, TE01] = np.sqrt(SIGNAL_POWER)

    def rates(z, amplitudes):
        """Return the interaction-picture rates at `z`."""
        field = np.einsum("bqcm,bm->bqc", electric, amplitudes * np.exp(-1j * beta * z))
        moduli = np.einsum("jqc,kqc->jkq", field, field.conj())
        squares = np.einsum("jqc,kqc->jkq", field, field)
        rate = np.zeros((n_bins, n_modes), dtype=complex)
        for out in range(n_bins):
            polarisation = np.zeros_like(field[0])
            for j in range(n_bins):
                for k in range(n_bins):
                    first = out - j + k
                    if 0 <= first < n_bins:
                        polarisation += 2 * moduli[j, k][:, None] * field[first]
                    second = j + k - out
                    if 0 <= second < n_bins:
                        polarisation += squares[j, k][:, None] * field[second].conj()
            polarisation *= EPS0 * chi3[:, None] / 4
            overlap = np.einsum(
                "qcm,qc,q->m", electric[out].conj(), polarisation, nodes.weights
            )
            rate[out] = (-1j * omega[out] * (inverse_flux[out] @ overlap) / 4) * np.exp(
                1j * beta[out] * z
            )
        return rate

    size = n_bins * n_modes

    def right_hand_side(z, packed):
        """Return the rates and their moduli, packed."""
        rate = rates(z, packed[:size].reshape(n_bins, n_modes))
        return np.concatenate([rate.ravel(), np.abs(rate).ravel().astype(complex)])

    solution = solve_ivp(
        right_hand_side,
        (0.0, length),
        np.concatenate([launch.ravel(), np.zeros(size, dtype=complex)]),
        method="DOP853",
        rtol=1e-12,
        atol=1e-12 * np.sqrt(PUMP_POWER),
    )
    assert solution.success
    final = solution.y[:, -1]
    return (
        final[:size].reshape(n_bins, n_modes),
        np.real(final[size:]).reshape(n_bins, n_modes),
        beta,
    )


ODE_HALF_WIDTH = 3
"""Comb bins the ODE carries on each side of the pump: seven in all.

The idler is one bin above the pump and the next cascaded products two and three;
the fourth is a product of four small amplitudes. The truncation is measured, not
assumed: the ODE is solved again at a half-width of two and the difference is a
term of the bound.
"""


@pytest.mark.slow
def test_vector_fwm(few_mode, record_property):
    """Intermodal four-wave mixing against the independent few-mode overlap ODE.

    An HE11-even pump and a TE01 signal one bin below it generate a TE01 idler one
    bin above, through the vector overlap of two pump fields with the signal's and
    the idler's; the pump also cross-couples TE01 to HE21 odd, which is in its
    class and near-degenerate with it, so the ODE and the route both carry a
    polarisation-rotating exchange as well as the idler. Every channel at every
    comb bin is compared, as a modulus, which neither channel gauge can change.

    **The bound, written before the comparison**, per channel and bin, is the sum
    of five measured terms:

    * the route's table against the provider at the comb's bins: its propagation
      constants differ by at most ``delta_beta`` and its fields and flux metric by
      at most ``v`` relative. A phase error in any of the four waves of a term moves
      that term by at most ``4 delta_beta L``, and a profile error its size by at
      most ``4 v``; each multiplies the **integrated rate modulus** of that channel
      and bin, which the ODE carries;
    * the route's own step error, twice the difference between ten and twenty
      fixed steps -- a Richardson estimate that holds for any order of at least
      one;
    * the ODE's truncation of the comb, the difference between half-widths three
      and two;
    * the ODE's own tolerance, 1e-10 of the pump amplitude, a hundred times its
      requested 1e-12 per step.
    """
    case = few_mode
    grid = case.grid
    state = cw_launch(case, (HE11_EVEN, TE01))
    bins = case.pump + np.arange(-ODE_HALF_WIDTH, ODE_HALF_WIDTH + 1)
    reference, integrated, beta = overlap_ode(case, bins, FWM_LENGTH)
    narrower, _, _ = overlap_ode(case, bins[1:-1], FWM_LENGTH)
    truncation = np.zeros_like(integrated)
    truncation[1:-1] = np.abs(np.abs(reference[1:-1]) - np.abs(narrower))
    truncation[[0, -1]] = np.abs(reference[[0, -1]])

    # The route's table against the provider at the comb's bins.
    delta_beta, profile = 0.0, 0.0
    for index, one in enumerate(bins):
        interpolated = case.table.evaluate(float(grid.omega[one]), 0.0)
        solved = case.provider.solve(float(grid.omega[one]), 0.0, nodes=case.nodes)
        delta_beta = max(
            delta_beta,
            float(
                np.max(np.abs(-np.imag(np.diag(interpolated.generator)) - beta[index]))
            ),
        )
        profile = max(
            profile,
            float(np.max(np.abs(interpolated.electric - solved.electric)))
            / float(np.max(np.abs(solved.electric))),
            float(np.max(np.abs(interpolated.flux_metric - solved.flux_metric))),
        )
    record_property("fwm_delta_beta_per_m", delta_beta)
    record_property("fwm_profile_departure", profile)

    coarse = run_fixed(case, state, FWM_STEPS)
    fine = run_fixed(case, state, 2 * FWM_STEPS)
    to_amplitude = 2.0 / grid.n_t
    route = np.abs(coarse[:, bins].T) * to_amplitude
    steps = 2.0 * np.abs(coarse[:, bins].T - fine[:, bins].T) * to_amplitude
    bound = (
        (4 * delta_beta * FWM_LENGTH + 4 * profile) * integrated
        + steps
        + truncation
        + 1e-10 * np.sqrt(PUMP_POWER)
    )
    departure = np.abs(route - np.abs(reference))
    record_property("fwm_worst_departure_over_bound", float(np.max(departure / bound)))
    idler = (ODE_HALF_WIDTH + 1, TE01)
    record_property("fwm_idler_ode", float(np.abs(reference[idler])))
    record_property("fwm_idler_route", float(route[idler]))
    record_property("fwm_idler_departure", float(departure[idler]))
    record_property("fwm_idler_bound", float(bound[idler]))
    # The process is what it claims to be: the idler is a sizeable fraction of the
    # signal, and it is not phase matched -- the mismatch turns through more than a
    # radian over the run, so the detuning is resolved, not neglected.
    mismatch = (
        2 * beta[ODE_HALF_WIDTH, HE11_EVEN]
        - beta[ODE_HALF_WIDTH - 1, TE01]
        - beta[ODE_HALF_WIDTH + 1, TE01]
    )
    record_property("fwm_mismatch_per_m", float(mismatch))
    assert abs(mismatch) * FWM_LENGTH > 1.0
    assert np.abs(reference[idler]) > 0.1 * np.sqrt(SIGNAL_POWER)
    assert np.all(departure <= bound)


# ------------------------------------------------ nonlinear mode exchange in a GRIN

GRIN_LENGTH = 1e-3
"""Length of every graded-index run, m: a millimetre."""

GRIN_BAND = (0.98e-6, 1.08e-6)
"""Declared band of the graded-index grids, m: wide enough for a 40 fs pulse."""

GRIN_ENERGY = 3e-8
"""Launched pulse energy, J: 30 nJ in 40 fs, about 0.7 MW of peak power.

Below silica's self-focusing power, and about half a radian of self-phase in LP01
over the millimetre, so the exchange is genuinely nonlinear and still perturbative.
"""

GRIN_SPLIT = (0.7, 0.3)
"""Fractions of the launched energy in LP01 and LP02; nothing in LP03."""

GRIN_BASE = {"elements": 32, "trange": 200e-15, "steps": 40}
"""The base of the ladder; each rung changes one of the three and keeps the others."""

GRIN_RUNGS = {
    "steps": (40, 80, 160),
    "elements": (32, 64, 128),
    "trange": (100e-15, 200e-15, 400e-15),
}
"""The three independent refinements: z step, radial elements, time window.

The z rungs start at forty steps over the millimetre because the order gate is a
statement about the **asymptotic** region. The fastest beat a Kerr term carries is
``2 (beta_LP01 - beta_LP03)``, about 3.2e4 1/m -- twice the two-group spacing of the
parabola -- and forty steps take 0.8 rad of it per step. A first ladder at 10, 20
and 40 steps, 3.2 rad per step at its coarsest, observed an order of 4.43 on LP03,
below the gate: the rungs were moved, the gate was not.
"""


def grin_grid_of(trange):
    """Return the graded-index field grid with a time window of `trange` s."""
    return FieldGrid.from_limits(GRIN_LENGTH, 1.03e-6, GRIN_BAND, trange)


def grin_knots():
    """Return the knots every graded-index guide of the ladder is sampled at.

    One set for all three time windows: the knots set the table's interpolation
    error, so a rung that moved them would refine the window and re-sample the
    table at once. They span the union of the three grids' active bands.
    """
    actives = [
        grin_grid_of(trange).omega[grin_grid_of(trange).active_mask]
        for trange in GRIN_RUNGS["trange"]
    ]
    low = min(float(active[0]) for active in actives)
    high = max(float(active[-1]) for active in actives)
    return np.linspace(0.99 * low, 1.01 * high, 15)


def grin_three_channels(elements):
    """Return the parabolic LP sector's first three radial channels, Kerr core.

    The parabolic profile of ``modal_provider_cases.grin_provider`` with a
    third channel, LP03, which is group five of the ideal parabola and close
    enough to cutoff at the active band's long end that the matching radius is
    three core radii rather than two.
    """
    return GradedIndexProvider(
        permittivity=PowerLawPermittivity(
            IndexMaterial(GRIN_AXIS_INDEX), GRIN_CONTRAST
        ),
        exterior=IndexMaterial(GRIN_AXIS_INDEX * np.sqrt(1.0 - 2.0 * GRIN_CONTRAST)),
        radius=GRIN_RADIUS,
        family="LP",
        count=3,
        elements=elements,
        extent=3.0,
        core_responses=(MaterialKerr(chi3=GENERAL_CHI3),),
    )


def grin_exit_energies(elements, trange, steps):
    """Return the three channels' exit energies of one rung of the ladder, J."""
    grid = grin_grid_of(trange)
    provider = grin_three_channels(elements)
    knots = grin_knots()
    guide = ModalGuide.from_provider(
        provider, knots, nodes=provider.node_rule(float(knots[0]), n_r=2)
    )
    route = general_modal_route(
        guide,
        grid,
        z_max=GRIN_LENGTH,
        omega_reference=float(grid.omega0),
        kerr_form="analytic",
    )
    pulse = GaussianPulse(wavelength=1.03e-6, fwhm=40e-15, energy=GRIN_ENERGY).build(
        grid
    )[0, 0]
    state = np.zeros((3, 1, grid.n_omega), dtype=complex)
    for channel, fraction in enumerate(GRIN_SPLIT):
        state[channel, 0] = np.sqrt(fraction) * pulse
    problem = assemble(
        grid,
        list(guide.mode_collection),
        [],
        state,
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[GRIN_LENGTH],
        dz_max=GRIN_LENGTH,
        dz_init=GRIN_LENGTH / steps,
        general=route,
        schedule=equal_steps(GRIN_LENGTH, steps),
    )
    owner = NumpyRefBackend(problem, grid=grid)
    try:
        owner.set_initial(problem.initial.field0, None)
        owner.propagate_to(GRIN_LENGTH)
        final = owner.observe().require_field()[0]
    finally:
        owner.close()
    return energy(grid, final)[:, 0]


def rung(name, value):
    """Return the ``(elements, trange, steps)`` of one rung of a refinement."""
    one = GRIN_BASE | {name: value}
    return (one["elements"], one["trange"], one["steps"])


@pytest.fixture(scope="module")
def grin_ladder():
    """Every rung of the three refinements, each run once."""
    runs = {}
    for name, values in GRIN_RUNGS.items():
        for value in values:
            key = rung(name, value)
            if key not in runs:
                runs[key] = grin_exit_energies(*key)
    return runs


def rung_differences(grin_ladder, name, channel):
    """Return one refinement's two successive differences in a channel's energy."""
    values = [grin_ladder[rung(name, value)][channel] for value in GRIN_RUNGS[name]]
    return abs(values[1] - values[0]), abs(values[2] - values[1]), values


@pytest.mark.filterwarnings("ignore::solprop.exceptions.GeneralModalBasisWarning")
@pytest.mark.slow
def test_grin_mode_exchange_converges(grin_ladder, record_property):
    """Nonlinear mode exchange in a GRIN guide, refined in z, elements and window.

    Section 9's "at least one nonlinear GRIN problem with mode/time/z refinement"
    (``test_multimode_soliton_convergence`` in its table). A 40 fs pulse is launched
    70/30 into LP01 and LP02 of the parabolic sector; the ideal parabola's groups
    are equally spaced, so ``2 LP02 -> LP01 + LP03`` is nearly phase matched and
    LP03, launched empty, fills through the Kerr term alone. **The observable** is
    the exit energy of LP03, which is purely nonlinear, and of LP01, which carries
    the exchange's other side.

    **The predicted behaviour, written before the ladder was run**, one rung at a
    time from the base (32 elements, 200 fs window, 40 steps over the millimetre):

    * **z step** (40, 80, 160 fixed steps): the fixed-step DOPRI5 is fifth order in
      the interaction picture, so the observed order must lie in the gate
      ``p - 0.5 <= observed <= p + 3``, with ``p = 5``;
    * **radial elements** (32, 64, 128): P1 elements put an ``h^2`` error in the
      propagation constants and the fields, so the same gate with ``p = 2``;
    * **time window** (100, 200, 400 fs): the window is periodic, and what a short
      one gets wrong is the overlap of the pulse's own tails across its edge,
      which falls like a Gaussian in the window length -- faster than any power,
      so the successive differences must fall by more than ``2^5`` and the finest
      must be below 1e-6 of the LP03 energy.

    The knots are the same on every rung (:func:`grin_knots`), so each rung refines
    one thing. The guide warns that its basis departs from orthonormal at the long
    end of the active band; the gate the route applies over the
    occupied bins passes on every rung, and the warning is not what is measured.
    """
    base = grin_ladder[rung("steps", GRIN_BASE["steps"])]
    for channel, label in ((2, "lp03"), (0, "lp01")):
        for name, rungs in GRIN_RUNGS.items():
            first, second, values = rung_differences(grin_ladder, name, channel)
            for value, energy_value in zip(rungs, values, strict=True):
                record_property(f"grin_ladder[{label},{name}={value:g}]", energy_value)
            record_property(f"grin_ladder_ratio[{label},{name}]", first / second)
    # The exchange is real: LP03 is fed, and far above anything a rung moves.
    assert base[2] > 1e-3 * GRIN_ENERGY
    for channel in (2, 0):
        first, second, _ = rung_differences(grin_ladder, "steps", channel)
        assert 4.5 <= np.log2(first / second) <= 8.0, (channel, first, second)
        first, second, _ = rung_differences(grin_ladder, "elements", channel)
        assert 1.5 <= np.log2(first / second) <= 5.0, (channel, first, second)
        first, second, _ = rung_differences(grin_ladder, "trange", channel)
        assert first / second > 2.0**5, (channel, first, second)
    _, finest, _ = rung_differences(grin_ladder, "trange", 2)
    assert finest < 1e-6 * base[2]

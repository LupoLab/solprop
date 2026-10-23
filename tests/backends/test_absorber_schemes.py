# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The temporal absorber as a per-step factor or inside the stage, compared.

The rule of the comparison (docs/dev/decisions.md) is accuracy and convergence first,
cost second, and that ``if (a) degrades the observed order [...] (b) becomes the default
whatever it costs``. It does, and the in-stage term is the default.

**The numbers, on the GNLSE N = 3 soliton launched into its temporal collar** (the case
below, at a sixteenth of a soliton period, against the same problem at 1600 fixed
steps, under a capped-logarithm temporal rate; the bounded profile the collar uses gives
the orders quoted in the tests below):

===============  ==========================================  ================
scheme           relative error at 100 / 200 / 400 steps     observed orders
===============  ==========================================  ================
``split_step``   7.657e-04  3.845e-04  1.912e-04              0.99  1.01
``nonlinear``    2.939e-05  1.278e-06  1.789e-08              4.52  6.16
===============  ==========================================  ================

and the energy the collar removes, from 100 steps to 400: ``split_step`` moves it by
6.5e-5 of itself, ``nonlinear`` by 9.0e-10.

**Why.** The split factor is exact *for the absorber alone* -- the factors telescope --
but applying it once per accepted step is a **Lie splitting** of a term that does not
commute with the linear flow, and a Lie splitting is first order. That caps the whole
scheme at one wherever the collar removes anything, and it is what makes the absorbed
energy drift with the step count. The in-stage term is part of the derivative the
Runge-Kutta method integrates, so it converges at the method's own order and the total
it removes stops depending on how the steps fell.

**What it costs**: one extra modal forward transform per stage; under a
capped-logarithm temporal rate, about 40 % more steps at equal tolerance on an adaptive
run (39 to 45 % measured), because the cap's rate reaches the error estimate -- the
bounded profile the collar uses returns the count to the split factor's -- and, the less
obvious finding, **conditional stability**. The collar's ceiling rate becomes an
eigenvalue of the linearised problem, so an explicit method needs ``h alpha / 2`` inside
its stability region. That is what :func:`solprop.boundaries.stable_step` returns and
what ``assemble`` warns about when a replayed schedule or a step floor is already past
it.
"""

import math
import warnings

import numpy as np
import pytest

from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.passes import apply_temporal_rate
from solprop.boundaries import STABILITY_RADIUS, absorber, stable_step
from solprop.exceptions import MaterialRangeWarning, TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.gnlse import problem_from
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SechPulse
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble

BETA2 = -11.83e-27  # s^2/m
GAMMA = 0.11  # 1/(W m)
TAU0 = 100e-15  # s
WAVELENGTH = 835e-9  # m
LENGTH = math.pi / 2 * TAU0**2 / abs(BETA2) / 16  # a sixteenth of a soliton period
ORDER = 3
# A collar five times stronger than the default, so that the boundary is doing real
# work over a distance this short rather than multiplying by one.
BOUNDARY_N = 4.0
REFERENCE_STEPS = 1600
SCHEMES = ("split_step", "nonlinear")
BOTH = (NumpyRefBackend, NumbaCpuBackend)


@pytest.fixture(scope="module")
def grid():
    """A window short enough that a soliton launched off centre reaches its collar."""
    return EnvelopeGrid.from_limits(LENGTH, WAVELENGTH, (500e-9, 2e-6), 1.2e-12)


def biting_case(grid, scheme, n_steps):
    """Assemble the soliton displaced towards the edge of the window."""
    fibre = problem_from(
        [0.0, 0.0, BETA2], omega_ref=grid.omega0, gamma=GAMMA, shock=False
    )
    peak = ORDER**2 * abs(BETA2) / (GAMMA * TAU0**2)
    built = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid)
    # A pure delay in the frequency domain: exp(i omega tau) shifts by tau, and a third
    # of the window puts the pulse most of the way into the collar.
    delay = 0.36 * (grid.t[-1] - grid.t[0])
    return assemble(
        grid,
        fibre.mode,
        [fibre.kerr],
        built * np.exp(1j * grid.omega * delay),
        controller=PIController(rtol=1e-10, atol=1e-20),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / n_steps,
        schedule=FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / n_steps, save_planes=[LENGTH]
        ),
        absorber=absorber(grid, boundary_N=BOUNDARY_N),
        absorber_scheme=scheme,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
    )


def exit_field(grid, scheme, n_steps):
    """Propagate the case on a fixed schedule and return the exit spectrum."""
    problem = biting_case(grid, scheme, n_steps)
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        return backend.observe().require_field()[0, 0, 0].copy()
    finally:
        backend.close()


@pytest.fixture(scope="module")
def exits(grid):
    """Every run the comparison needs, computed once: 3.5 s of the medium tier.

    Both schemes at 100, 200 and 400 fixed steps, and both at the reference count. The
    tests below read this table rather than each propagating for itself, because the
    same runs answer all four questions the comparison asks.
    """
    counts = (100, 200, 400, REFERENCE_STEPS)
    return {
        (scheme, n): exit_field(grid, scheme, n) for scheme in SCHEMES for n in counts
    }


def energy_of(grid, spectrum):
    """Total energy of a spectrum, J."""
    field = np.fft.ifft(spectrum, axis=-1) * grid.n_t
    return float(np.sum(np.abs(field) ** 2) * grid.dt)


def relative(got, want):
    """Relative L2 deviation of two spectra."""
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


@pytest.mark.medium
def test_the_collar_bites_on_this_case(grid, exits):
    """Otherwise every comparison below would be about a boundary that does nothing."""
    problem = biting_case(grid, "nonlinear", 400)
    start = energy_of(grid, problem.initial.field0[0, 0, 0])
    left = energy_of(grid, exits[("nonlinear", 400)])
    # A third of the pulse walks into the boundary and is absorbed over this distance
    # (a seventh under the capped logarithm, whose inner collar was almost transparent).
    assert 0.05 < (start - left) / start < 0.5


@pytest.mark.medium
@pytest.mark.parametrize(
    ("scheme", "lowest", "highest"),
    [("split_step", 0.8, 1.3), ("nonlinear", 3.5, 8.0)],
    ids=SCHEMES,
)
def test_only_the_in_stage_absorber_keeps_the_schemes_order(
    exits, scheme, lowest, highest
):
    """The split factor is a Lie splitting and is first order; the term is not.

    Measured 1.00 and 1.00 against 4.02 and 4.95 under the bounded temporal profile
    (0.99 and 1.01 against 4.52 and 6.16 under a capped-logarithm rate); the gates are
    those with room for the round-off floor the finer pair is approaching, and they do
    not overlap.
    """
    truth = exits[("nonlinear", REFERENCE_STEPS)]
    errors = [relative(exits[(scheme, n)], truth) for n in (100, 200, 400)]
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert all(lowest < one < highest for one in orders), orders


@pytest.mark.medium
def test_only_the_in_stage_absorber_removes_a_step_count_independent_energy(
    grid, exits
):
    """The comparison's second criterion, and the one a per-step window fails.

    Quadrupling the steps moves what the split factor removes by 5.9e-5 of itself --
    first order, like everything else about it -- and what the in-stage term removes by
    2.3e-13, which is the round-off of the reduction (6.5e-5 and 9.0e-10 under a
    capped-logarithm rate).
    """
    drifts = {}
    for scheme in SCHEMES:
        coarse, fine = (energy_of(grid, exits[(scheme, n)]) for n in (100, 400))
        drifts[scheme] = abs(coarse - fine) / fine
    assert drifts["split_step"] > 1e-5
    assert drifts["nonlinear"] < 1e-8
    assert drifts["nonlinear"] < 1e-4 * drifts["split_step"]


@pytest.mark.medium
def test_the_two_schemes_agree_once_both_are_converged(exits):
    """They discretise the same equation, so the disagreement is one of them being
    wrong.

    At 100 steps they differ by 1.7e-3 and at 1600 by 1.1e-4, and it is the split
    factor that is away from the limit -- the in-stage term is at 7e-12 by 400 steps;
    refine the factor far enough and it comes back (7.7e-4 at 100 steps under the
    capped logarithm, whose gentler inner collar made the splitting error smaller).
    """
    truth = exits[("nonlinear", REFERENCE_STEPS)]
    assert relative(exits[("split_step", REFERENCE_STEPS)], truth) < 2e-4
    assert relative(exits[("nonlinear", 400)], truth) < 1e-7


def test_a_schedule_too_long_for_an_in_stage_collar_is_refused_before_the_run(grid):
    """The in-stage term's stiffness, refused rather than diverged at.

    An adaptive run finds the limit for itself, because the rate reaches its ceiling
    only where the field is negligible and an enormous derivative on an empty bin
    scores nothing in the error norm. A replayed schedule cannot, so `assemble`
    refuses it, naming the limit; the split factor integrates the same term exactly
    and has no such limit.
    """
    rate = absorber(grid, boundary_N=BOUNDARY_N).temporal_rate
    limit = stable_step(rate)
    assert limit == pytest.approx(2.0 * STABILITY_RADIUS / float(np.max(rate)))
    too_few = max(int(LENGTH / limit) // 2, 1)
    with pytest.raises(TranslationError, match=f"only up to {limit:.3e} m"):
        biting_case(grid, "nonlinear", too_few)
    # Steps inside the limit are accepted, and the same too-long schedule under the
    # split factor is unconditionally stable; neither says anything.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        biting_case(grid, "nonlinear", math.ceil(LENGTH / limit))
        biting_case(grid, "split_step", too_few)


def test_a_step_floor_above_the_limit_is_refused_too(grid):
    """The other configuration that cannot find the limit for itself."""
    rate = absorber(grid, boundary_N=BOUNDARY_N).temporal_rate
    fibre = problem_from(
        [0.0, 0.0, BETA2], omega_ref=grid.omega0, gamma=GAMMA, shock=False
    )
    peak = ORDER**2 * abs(BETA2) / (GAMMA * TAU0**2)
    built = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid)
    with pytest.raises(TranslationError, match="the step floor dz_min"):
        assemble(
            grid,
            fibre.mode,
            [fibre.kerr],
            built,
            controller=PIController(rtol=1e-10, atol=1e-20),
            save_planes=[LENGTH],
            dz_max=LENGTH,
            # The floor and the first step together: with the bounded profile the
            # limit is long enough that four of it passes a quarter of the length.
            dz_init=4.0 * stable_step(rate),
            dz_min=4.0 * stable_step(rate),
            absorber=absorber(grid, boundary_N=BOUNDARY_N),
            absorber_scheme="nonlinear",
            density=1.0,
            shock=False,
            prefactor_index=1.0,
        )


# --------------------------------------------- a collar with nothing else to evaluate


def collar_only(grid, scheme):
    """The linear flow and a temporal collar: no response at all."""
    fibre = problem_from(
        [0.0, 0.0, BETA2], omega_ref=grid.omega0, gamma=GAMMA, shock=False
    )
    peak = ORDER**2 * abs(BETA2) / (GAMMA * TAU0**2)
    built = SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid)
    delay = 0.36 * (grid.t[-1] - grid.t[0])
    return assemble(
        grid,
        fibre.mode,
        [],
        built * np.exp(1j * grid.omega * delay),
        controller=PIController(rtol=1e-9, atol=1e-20),
        save_planes=[LENGTH],
        dz_max=LENGTH / 8,
        dz_init=LENGTH / 400,
        absorber=absorber(grid, boundary_N=BOUNDARY_N),
        absorber_scheme=scheme,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
    )


@pytest.mark.parametrize("backend_class", BOTH, ids=("numpy_ref", "numba_cpu"))
def test_a_collar_with_no_response_is_still_absorbed(grid, backend_class):
    """The one case the stage's short circuit would drop, on both backends.

    A problem whose only compiled entry is the accumulation has an identically zero
    polarisation, and both backends skip the transforms rather than multiply by nothing.
    The in-stage absorber is a term of its own and is not part of the polarisation, so
    the short circuit has to keep it: without that, a linear run with a collar would
    silently absorb nothing at all.
    """
    energies = {}
    for scheme in SCHEMES:
        problem = collar_only(grid, scheme)
        backend = backend_class(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(LENGTH)
            energies[scheme] = energy_of(
                grid, backend.observe().require_field()[0, 0, 0]
            )
        finally:
            backend.close()
        start = energy_of(grid, problem.initial.field0[0, 0, 0])
        assert 0.05 < (start - energies[scheme]) / start < 0.5, scheme
    # The two schemes absorb the same energy, to the first-order gap between them.
    assert energies["nonlinear"] == pytest.approx(energies["split_step"], rel=1e-3)


# ------------------------------------------- the accepted state stays inside the band

BAND_LENGTH = 2e-3  # m
BAND_STEPS = 5


def capillary_in_its_collar(scheme):
    """A field-grid capillary pulse delayed into a strong collar, five fixed steps."""
    grid = FieldGrid.from_limits(BAND_LENGTH, 800e-9, (200e-9, 4e-6), 100e-15)
    mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(grid)
    delay = 0.4 * (grid.t[-1] - grid.t[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        problem = assemble(
            grid,
            mode,
            [KerrResponse(chi3=2.2e-52, thg=True)],
            start * np.exp(1j * grid.omega * delay),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[BAND_LENGTH],
            dz_max=BAND_LENGTH,
            dz_init=BAND_LENGTH / BAND_STEPS,
            schedule=FixedSchedule.uniform(
                0.0, BAND_LENGTH, BAND_LENGTH / BAND_STEPS, save_planes=[BAND_LENGTH]
            ),
            absorber=absorber(grid, boundary_N=BOUNDARY_N),
            absorber_scheme=scheme,
        )
    return grid, problem, BAND_LENGTH


def soliton_in_its_collar(grid, scheme):
    """The envelope soliton of :func:`biting_case`, on five fixed steps."""
    return grid, biting_case(grid, scheme, BAND_STEPS), LENGTH


@pytest.mark.parametrize("kind", ["field", "envelope"])
def test_the_split_step_absorber_leaves_nothing_outside_the_active_band(
    grid, backend_class, kind
):
    """The accepted state is exactly zero outside the active band, the DC included.

    The split factor is a product in time, which spreads content into every frequency
    bin; on a field grid the real field's DC bin is the largest of them. The band is
    part of the definition of the problem (``docs/guide/numerics/boundaries.md``), so
    the mask
    has to follow the absorber, as it does the in-stage term: before the fix the
    reference kept a DC of 1 % of the spectral peak on the field case below.
    """
    if kind == "field":
        run_grid, problem, length = capillary_in_its_collar("split_step")
    else:
        run_grid, problem, length = soliton_in_its_collar(grid, "split_step")
    backend = backend_class(problem, grid=run_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(length)
        field = np.asarray(backend.observe().require_field())
    finally:
        backend.close()
    outside = ~problem.grids.active_mask
    assert outside.any()
    assert np.all(field[..., outside] == 0)
    assert np.abs(field).max() > 0
    if kind == "field":
        # The test bites: one more factor alone puts a DC of order a percent of the
        # peak back (the envelope's spill is 6e-14 of it, which only the exact zero
        # above can see).
        rate = problem.responses.temporal_absorber
        assert rate is not None
        absorbed = apply_temporal_rate(field, rate, length / BAND_STEPS, kind=kind)
        assert outside[0]
        assert np.abs(absorbed[..., 0]).max() > 1e-3 * np.abs(absorbed).max()


def test_the_split_step_absorber_agrees_with_the_reference(equivalence_backend_class):
    """The compiled backends follow the reference through the split factor and its mask.

    Kokkos always committed the factor through the band mask; while the reference did
    not, the two differed by the 1 % DC on this case and by what that DC did to the
    stages after it.
    """
    fields = []
    for backend_class in (NumpyRefBackend, equivalence_backend_class):
        run_grid, problem, length = capillary_in_its_collar("split_step")
        backend = backend_class(problem, grid=run_grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(length)
            fields.append(np.asarray(backend.observe().require_field()))
        finally:
            backend.close()
    reference, compiled = fields
    assert np.abs(compiled - reference).max() <= 1e-13 * np.abs(reference).max()

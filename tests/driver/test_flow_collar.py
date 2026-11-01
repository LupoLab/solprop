# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled flow's explicit residual outside the scored band.

A ``"frozen_residual"`` flow carries the part of the generator its section's frozen
block leaves out, ``R_s(z) = G(z) - C(z) - G_s``, as an explicit term of the
interaction-picture Runge-Kutta source. That term is applied only on the requested
band, the grid's ``inband_mask``; in the spectral collar outside it -- still inside the
active band the wall block is evaluated on, and scored by the error norm since kernel
schema 23 -- the frozen flow alone carries the state
(``docs/guide/waveguides/capillaries.md``).
Applied there, where the wall model is outside its validity, the residual can grow
towards the end of a long frozen section. The bound on what dropping it costs, in
short:

* the linear flow is diagonal in frequency and the hard mask is per bin, so an in-band
  bin reads the collar only through the source's frequency mixing -- the nonlinear
  response and the in-stage temporal absorber, with Lipschitz scale ``Lambda``;
* dropping the residual moves the collar by at most ``C K_c z`` of its content
  ``sigma_c`` (``K_c`` the collar's declared residual norm, ``C`` the certificate's
  conditioning), so the in-band answer moves by at most
  ``Gamma exp(Gamma Lambda L) (Lambda L) (C K_c L / 2) sigma_c`` relative, with
  ``Gamma = C^S exp(Phi)`` the in-band growth of the parity bound
  (``tests/flow_parity.py``).

A problem whose requested band covers every bin applies the residual everywhere, which
is how these tests reach that route without keeping separate code for it.
"""

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

import flow_parity
from driver.test_coherent_capillary import (
    GRID,
    LENGTH,
    STEPS,
    VECTOR_WALL,
    WEAK_COLLAR,
    taper,
    varying,
    vector_modes,
)
from flow_parity import (
    ENERGY,
    QUIET,
    apply_bound,
    build,
    flow_scales,
    formation_bound,
    run,
)
from legacy_geometry import LEGACY_GRID
from march_bounds import FIXED_ORDER
from modal_provider_cases import GENERAL_Z_END
from solprop.backends import available
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import Absorber, absorber
from solprop.driver import Simulation
from solprop.exceptions import SolpropWarning
from solprop.grid import FieldGrid
from solprop.modes import CoherentWall
from solprop.profiles import PiecewiseLinear
from solprop.stepper import FixedSchedule
from solprop.translate.coupled import flow_section, frozen_generator
from solprop.translate.problem import KernelProblem
from state_records import check_state, record_bound

pytestmark = [
    pytest.mark.filterwarnings(f"ignore::{category.__module__}.{category.__name__}")
    for category in QUIET
] + [pytest.mark.filterwarnings("ignore:the temporal absorber attenuates")]

BACKENDS = ("numpy_ref", "numba_cpu", "native")
"""The three CPU backends, which must all leave the collar to the frozen flow."""

UNIT = 2.0**-53
"""The unit roundoff of float64."""


def backend_class(name):
    """Return one backend's class, skipping when it is not installed."""
    if name not in available():
        pytest.skip(f"the {name} backend is not installed")
    if name == "native":
        # Optional backend: absent from the base environment.
        from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            NativeBackend,
        )

        return NativeBackend
    return {"numpy_ref": NumpyRefBackend, "numba_cpu": NumbaCpuBackend}[name]


def scored_everywhere(problem: KernelProblem) -> KernelProblem:
    """Return the problem with every bin scored: the residual applied everywhere."""
    grids = problem.grids
    return replace(
        problem,
        grids=replace(grids, inband_mask=np.ones_like(grids.inband_mask)),
    )


def collar_bins(problem: KernelProblem):
    """Return the flow's bins the error norm does not score, bool over the bins."""
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.generator is not None
    active = np.any(flow.generator != 0.0, axis=(0, 1, 2, 3))
    return active & ~problem.grids.inband_mask


# ------------------------------------------------------------- the collar's route


def _no_temporal_collar():
    """Return the weak spectral collar with the temporal one removed.

    The temporal absorber is the one linear term that mixes frequencies; without it,
    and at a launch of one bin at a nanowatt, a collar bin evolves by its own linear
    flow alone, to rounding.
    """
    return Absorber(
        spectral_rate=WEAK_COLLAR.spectral_rate,
        temporal_rate=np.zeros_like(WEAK_COLLAR.temporal_rate),
        temporal_rate_fine=np.zeros_like(WEAK_COLLAR.temporal_rate_fine),
        band_mask=WEAK_COLLAR.band_mask,
        reference_length=WEAK_COLLAR.reference_length,
    )


def _launched_in(problem: KernelProblem, index: int) -> KernelProblem:
    """Return the problem launched with 1e-9 in the first channel at one bin only."""
    state = np.zeros_like(problem.initial.field0)
    state[0, 0, 0, index] = 1e-9
    return replace(problem, initial=replace(problem.initial, field0=state))


def _frozen_product(problem: KernelProblem, index: int):
    """Return ``prod_k exp(dz G_{s_k})`` at one bin, the frozen flow's own answer."""
    flow = problem.linear.coupled_flow
    assert flow is not None
    schedule = problem.stepper.schedule
    assert schedule is not None
    product = np.eye(flow.n_modes, dtype=complex)
    anchor = 0.0
    for step in schedule.steps:
        block = frozen_generator(flow, flow_section(flow, anchor))[:, :, index]
        product = expm(float(step) * block) @ product
        anchor += float(step)
    return product


@pytest.mark.medium
@pytest.mark.parametrize("name", BACKENDS)
def test_the_collar_is_carried_by_the_frozen_flow_alone(name):
    """A collar bin propagates by ``exp(dz G_s)`` per step, the residual left out.

    Launched in `HE_11` at the lowest active bin (3.18 um, below the scored band's
    1.6 um) of the qualification taper, with no temporal collar and at a nanowatt, so
    that nothing mixes frequencies: the channels' amplitudes relative to `HE_11`'s,
    which the channel-common diagonal leaves alone, are the product of the frozen
    sections' exponentials over the ten steps. Bound, stated first: each action is
    within the formation bound ``eps_form`` of ``tests/flow_parity.py`` of the exact
    exponential, with ``eps_apply``'s rounding of its product, ten of them compose,
    carried by the certificate's conditioning per step, and a ratio of two amplitudes
    doubles it: ``2 * 10 * C^10 * apply_bound``. Scoring every bin puts the residual
    back, which moves the same ratios by more than 1e-3 (the connection converts power
    there too), so the comparison is not blind to it. Measured 3.4e-16 on every
    backend, bound 4.6e-13.
    """
    backend = backend_class(name)
    problem = varying(wall=CoherentWall(), absorber=_no_temporal_collar()).translate()
    index = int(np.flatnonzero(collar_bins(problem))[0])
    assert not GRID.inband_mask[index]
    launched = _launched_in(problem, index)
    expected = _frozen_product(problem, index)[:, 0]
    driver = backend(launched, grid=GRID)
    try:
        driver.set_initial(launched.initial.field0, None)
        driver.propagate_to(LENGTH)
        got = driver.observe().require_field()[0, :, 0, index].copy()
    finally:
        driver.close()
    flow = problem.linear.coupled_flow
    assert flow is not None
    condition = float(np.max(flow.inverse_condition_bound))
    dz = LENGTH / STEPS
    bound = (
        2.0
        * STEPS
        * condition**STEPS
        * apply_bound(flow.n_modes, condition, formation_bound(problem, dz))
    )
    departure = float(np.max(abs(got / got[0] - expected / expected[0])))
    assert departure <= bound, (departure, bound)
    if name == "numpy_ref":
        old = run(NumpyRefBackend, scored_everywhere(launched), GRID)[0, :, 0, index]
        assert float(np.max(abs(old / old[0] - expected / expected[0]))) > 1e-3


# ------------------------------------- the collar transient in miniature

TAPER_LENGTH = 0.049
"""Length of the single frozen section, m: 100 to 60 um, the radius spread of the
worked tapered-capillary example's 20 cm section, so that its residual at 5.69 um
reaches that section's 7.3e3 1/m."""

TRANSIENT_STEP = 7e-4
"""Fixed step, m: ``K dz`` of 5.1 at the collar's worst bin, where DOPRI5's stability
function on the imaginary axis is far above one; 2.7 in band."""


def _transient_case():
    """Return the miniature: a 4.9 cm taper and ten steps of straight bore after it."""
    total = TAPER_LENGTH + STEPS * TRANSIENT_STEP
    count = round(total / TRANSIENT_STEP)
    grid = FieldGrid.from_limits(total, 800e-9, (300e-9, 3e-6), 250e-15)
    with warnings.catch_warnings():
        # The advisory-step warning is the point of the case: its step is above the
        # residual's advisory limit on purpose.
        for category in (*QUIET, SolpropWarning):
            warnings.simplefilter("ignore", category)
        simulation = Simulation.from_capillary(
            PiecewiseLinear.from_stages(
                (0.0, TAPER_LENGTH, count * TRANSIENT_STEP), [100e-6, 60e-6, 60e-6]
            ),
            count * TRANSIENT_STEP,
            "He",
            1.5,
            wall=CoherentWall(),
            modes=4,
            grid=grid,
            absorber=absorber(grid, boundary_length=1.0e3),
            wavelength=800e-9,
            fwhm=10e-15,
            energy=1e-9,
            dz_max=TRANSIENT_STEP,
            dz_init=TRANSIENT_STEP,
            schedule=FixedSchedule(np.full(count, TRANSIENT_STEP)),
            n_save=2,
            backend="numpy_ref",
            plasma=False,
            raman=False,
            noise=None,
        )
        return simulation.translate(), grid


@pytest.fixture(scope="module")
def transient():
    """Build the miniature once for the module (about a second)."""
    return _transient_case()


def _seeded_growth(backend, problem, grid, collar, worst):
    """Return `HE_14`'s growth at the worst collar bin over the ten steps to the corner.

    The launch is restarted ten steps before the corner, as a probe of the full-length
    run restarted it at 0.683 m, with 1e-8 of the launch's peak seeded in every channel
    at every collar bin.
    """
    launch = problem.initial.field0.copy()
    seed = 1e-8 * float(np.max(abs(launch)))
    launch[..., collar] += seed
    start = TAPER_LENGTH - STEPS * TRANSIENT_STEP
    driver = backend(problem, grid=grid)
    try:
        driver.set_initial(launch, None, z=start)
        driver.propagate_to(TAPER_LENGTH)
        final = driver.observe().require_field()
    finally:
        driver.close()
    return float(abs(final[0, 3, 0, worst])) / seed


@pytest.mark.parametrize("name", BACKENDS)
def test_a_coupled_flow_keeps_its_collar_in_every_stage(transient, name):
    """A quiet collar is never left out beside a coupled flow, on any backend.

    The frozen flow carries the collar (SEP 21 D15), so the term stays in every
    stage whatever the monitor says: one rule, decided by the driver, for all three
    backends (the native engine kept it and the Python ones dropped it before).
    """
    backend = backend_class(name)
    problem, grid = transient
    assert problem.linear.coupled_flow is not None
    driver = backend(problem, grid=grid)
    try:
        assert not driver._collar_can_rest()
        driver._set_collar_quiet(True)
        assert not driver._collar_quiet
    finally:
        driver.close()


@pytest.mark.medium
@pytest.mark.parametrize("name", BACKENDS)
def test_a_seeded_collar_no_longer_grows_towards_a_frozen_section_s_end(
    transient, name
):
    """The collar transient of a long frozen section, reproduced in ten steps, and gone.

    At the end of a long frozen section the collar's residual reaches 7.3e3 1/m at
    5.69 um, and ten 0.7 mm steps put ``K dz`` at 5.1 there: with the residual
    applied (every bin scored), `HE_14` at that bin grows by more than 1e3 by the
    corner (measured 1.9e5; 1e5 in the full-length run), unseen by the in-band error.
    Without it the bin evolves by the frozen flow alone, whose growth over the ten steps
    is at most the product of ``||exp(dz G_s)||_2`` at that bin (computed here, 1.11):
    the bound, with 1e-6 for the Kerr source at a nanojoule and the weak temporal collar
    (measured 7.8e-2 on all three backends).
    """
    backend = backend_class(name)
    problem, grid = transient
    flow = problem.linear.coupled_flow
    assert flow is not None
    collar = collar_bins(problem)
    rows = flow.residual_norm_bound[:, collar]
    worst = int(np.flatnonzero(collar)[int(np.argmax(np.max(rows, axis=0)))])
    assert float(np.max(rows)) * TRANSIENT_STEP > 4.0
    frozen = 1.0
    start = TAPER_LENGTH - STEPS * TRANSIENT_STEP
    for step in range(STEPS):
        block = frozen_generator(
            flow, flow_section(flow, start + step * TRANSIENT_STEP)
        )[:, :, worst]
        frozen *= float(np.linalg.norm(expm(TRANSIENT_STEP * block), 2))
    growth = _seeded_growth(backend, problem, grid, collar, worst)
    assert growth <= frozen * (1.0 + 1e-6), (growth, frozen)
    old = _seeded_growth(backend, scored_everywhere(problem), grid, collar, worst)
    assert old > 1e3, old


# ----------------------------------------------------------- the in-band answer


def _vector_taper():
    """Return the TE/TM wall on the qualification taper at the parity energy."""
    return varying(wall=VECTOR_WALL, modes=vector_modes(radius=taper()), energy=ENERGY)


IN_BAND_CASES = {
    "taper": lambda: build("taper"),
    "gradient": lambda: build("gradient"),
    "te_tm_taper": _vector_taper,
}
"""The varying coupled-flow cases, each with a Kerr half strong enough to couple."""


def in_band_bound(problem: KernelProblem, grid, planes):
    """Return the derived relative bound on the in-band answer's move, and its parts.

    ``planes`` are the new run's states after each of its steps, the last the answer.
    """
    flow = problem.linear.coupled_flow
    assert flow is not None
    scales = flow_scales(problem, grid, planes[-1])
    collar = collar_bins(problem)
    residual = float(np.max(flow.residual_norm_bound[:, collar], initial=0.0))
    content = max(
        float(np.linalg.norm(plane[..., collar])) for plane in planes
    ) / float(np.linalg.norm(planes[-1]))
    temporal = problem.responses.temporal_absorber
    rate = 0.0 if temporal is None else float(np.max(temporal))
    coupling = 3.0 * scales.phi + 0.5 * rate * LENGTH
    growth = scales.condition**STEPS * math.exp(scales.phi)
    analytic = (
        growth
        * math.exp(growth * coupling)
        * coupling
        * (scales.condition * residual * LENGTH / 2.0)
        * content
    )
    return analytic + growth * 2.0 * STEPS * UNIT, analytic, content


@pytest.mark.medium
@pytest.mark.parametrize("case", sorted(IN_BAND_CASES))
def test_the_in_band_answer_moves_by_less_than_the_derived_bound(case):
    """Dropping the collar's residual moves the scored band by at most the bound.

    Bound, derived before the run (module docstring): the in-band
    move is the product of the source's coupling ``Lambda L`` (three times the
    launch's nonlinear phase plus half the temporal collar's largest rate times the
    length), the dropped term ``C K_c L / 2`` and the collar's largest share of the
    field over the steps, carried by ``Gamma = C^S e^Phi``; plus one rounding of the
    accept and one of the stage input per step, ``2 S u Gamma``, for a changed source
    tipping a rounding. The collar holds rounding here (6.0e-16 of the field), so the
    analytic part is below 1e-17 (7.8e-19 on the taper, 6.4e-19 on the TE/TM taper,
    1.0e-24 under the gradient) and the answer moves by less than it (measured 5.8e-20,
    3.0e-20 and 9.0e-28), far inside the whole bound (2.5e-15, set by the rounding
    floor).
    """
    simulation = IN_BAND_CASES[case]()
    problem = simulation.translate()
    grid = simulation.grid
    band = problem.grids.inband_mask
    planes = []
    driver = NumpyRefBackend(problem, grid=grid)
    try:
        driver.set_initial(problem.initial.field0, None)
        for step in range(STEPS):
            driver.propagate_to((step + 1) * LENGTH / STEPS)
            planes.append(driver.observe().require_field().copy())
    finally:
        driver.close()
    old = run(NumpyRefBackend, scored_everywhere(problem), grid)
    departure = float(
        np.linalg.norm((planes[-1] - old)[..., band]) / np.linalg.norm(old[..., band])
    )
    bound, analytic, content = in_band_bound(problem, grid, planes)
    assert content > 0.0
    assert analytic < 1e-17, analytic
    assert departure <= bound, (departure, bound)


# ------------------------------------------------- unchanged where nothing moves

# These were byte digests of each case's final state, recorded on one platform and
# re-recorded at every deliberate change (the Multistep default, SEP 3's optmat helium
# and silica wall, the wall's refractivity difference, the rationalised refractivity
# root, the 293.15 K default temperature, the general route's symmetric contraction);
# the git history keeps them. They are now stored final states of the fixed-order march,
# compared within its rounding budget (tests/state_records.py), which a fresh
# environment, another build of a dependency or a reordered kernel does not exceed.


@pytest.mark.medium
@pytest.mark.parametrize("name", ["numpy_ref", "numba_cpu"])
@pytest.mark.parametrize("case", ["scalar", "te_tm", "split", "taper", "gradient"])
def test_a_flow_with_nothing_unscored_is_what_it_was(
    case, name, ndarrays_regression, monkeypatch
):
    """A constant section, or a varying one with every bin scored, is unchanged.

    The constant cases take no residual at all; a varying case whose every active
    bin is scored applies it where it always did, in the same order. The final state
    after ten fixed steps at the fixed order matches its record within the march's
    rounding budget (``tests/state_records.py``), on the grids it was recorded on:
    the parity cases' grids in the former geometry, with their weak collar.
    """
    grid = FieldGrid.from_limits(
        LENGTH, 800e-9, (400e-9, 1.6e-6), 300e-15, **LEGACY_GRID
    )
    monkeypatch.setattr(flow_parity, "GRID", grid)
    monkeypatch.setattr(
        flow_parity, "WEAK_COLLAR", absorber(grid, boundary_length=1.0e3)
    )
    monkeypatch.setattr(
        flow_parity,
        "general_grid",
        lambda: FieldGrid.from_limits(
            GENERAL_Z_END, 835e-9, (815e-9, 860e-9), 60e-15, **LEGACY_GRID
        ),
    )
    backend = backend_class(name)
    simulation = build(case, integrator=FIXED_ORDER)
    problem = simulation.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    if case in {"taper", "gradient"}:
        assert flow.section_methods != ("constant_schur",)
        problem = scored_everywhere(problem)
    else:
        assert flow.section_methods == ("constant_schur",)
    state = run(backend, problem, simulation.grid)
    check_state(ndarrays_regression, state, record_bound(simulation, state))

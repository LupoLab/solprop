# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage march's parity against the reference: SEP 24 7.2's bound and runs.

Each parity case replays one schedule of at most ten steps on the reference and on a
compiled backend, and the states agree within a rounding bound fixed before the
comparison (SEP 25; :func:`parity_bound`): no agreement is bitwise. The rows are the
case catalogue's (``backends/session_rows.py``). The march itself (:data:`MARCH`,
:func:`marched`) and its Cartesian Kerr beam (:func:`cartesian_march`) are here too,
for the reference's own tests (``test_low_storage.py``) and native's.
"""

import dataclasses
import itertools
import math

import numpy as np

from bounds import EPS, assert_within, departure, step_budget
from legacy_geometry import with_collar_rate
from sessions.runs import run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import ExecutionOptions, Start
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, PIController
from solprop.translate.assembly import assemble
from solprop.translate.linop import delta_phi
from solprop.translate.problem import Translation

MARCH = LowStorageRungeKutta()
"""The low-storage march, SEP 24's RK4IP."""

# Section 4.5's argument coefficient, fixed before any measurement. Each flow's exponent
# and the reference's carry about eps |Delta Phi|; the table is built from its key, the
# run's mean step (Q10), so a reused flow is off by |D| |h - h_t| / 2, a few ulps of the
# position times |D|, whose sum over a run telescopes to the run's own phase (the
# |Phi(z_N)| term); and the products add 4 N eps. 16 covers the 8 + 1 of phase 1's
# per-step key with a factor of about two.
ARGUMENT_FACTOR = 16


def marched(problem, run_request, schedule, *, mode="auto", **stepper):
    """Return a problem and its request stepped by the low-storage march."""
    settings = dataclasses.replace(
        run_request.stepper,
        integrator=MARCH,
        schedule=schedule,
        propagator_mode=mode,
        dz_max=max(schedule.max_step, run_request.stepper.dz_max),
        dz_init=min(run_request.stepper.dz_init, schedule.max_step),
        **stepper,
    )
    return Translation(problem, run_request.with_stepper(settings))


def populated(spectra, fraction=1e-2):
    """Return the bins holding at least `fraction` of the run's peak amplitude."""
    peak = np.max(np.abs(np.asarray(spectra)), axis=0)
    return peak >= fraction * np.max(peak)


CARTESIAN_CHUNK = ExecutionOptions(node_chunk=7)
"""The node chunk :func:`cartesian_march`'s problem runs with: 7 of 108 columns."""


def cartesian_march(mode, *, noise=None, collar=False):
    """Return a Cartesian Kerr problem under the march, run in chunks of 7 of 108.

    Its backends take :data:`CARTESIAN_CHUNK`.

    With the collars, the spatial and the temporal one in-stage, it takes the 100
    steps their stability limits ask for; otherwise 4.
    """
    length = 1e-3
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0.25 if collar else 0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    start = (
        pulse.build(grid, space, medium=SpatialMedium())
        if noise is None
        else np.zeros_like(noise.spectrum)
    )
    schedule = FixedSchedule.uniform(0.0, length, length / (100 if collar else 4))
    problem, run_request = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        noise=noise,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
        propagator_mode=mode,
        absorber=absorber(grid) if collar else None,
        absorber_scheme="nonlinear",
    )
    if collar:
        # A collar cannot be designed on this coarse box (SEP 40's floor); the subject
        # is the march's ledger beside an in-stage term, so its rate is put on directly.
        problem = with_collar_rate(problem, 12.6 / (length / 20))
    return problem, run_request, grid, space


STEPS = 10
"""The qualification scope's propagation (AGENTS.md): ten steps."""

PRODUCTS = 32
"""Roundings of one evaluation outside its transforms, with a margin: the input's add
and scale, the prefactor, the window and the absorber's term, the fold's four products
and adds, the flow's exponential (a few eps), and the response's own few per sample."""

FFT_CONSTANT = 8
"""Roundings per level of a transform pair, Higham's ``log2 n`` with a margin."""


def parity_bound(problem, run_request, *, growth=2.0, transverse=1, depth=0):
    """Return the relative bound two backends' replays of one schedule must meet.

    Fixed before either is run, from the case's own sizes:

    - **the steps' rounding**, ``growth N 4 (PRODUCTS + FFT_CONSTANT log2(n_to n_x n_y)
      + depth) u``: four evaluations a step, each a temporal transform pair over
      ``n_to`` samples, a transverse pair over the free kinds' ``n_x n_y`` columns, a
      response whose cumulative sums have addition depth `depth` (the plasma's prefix
      over ``n_to``), and the fold;
    - **the flows' arguments**, section 4.5's ``c eps (N |D| h + |Phi(z_N)|)`` over the
      bins holding a thousandth of the launch's peak, since the backends form
      ``exp(h D / 2)`` from differently ordered exponents (the table on both, or the
      separable body against the reference's propagator), carried by the same growth;
      on the separable kinds the exponent is a principal root's, one ulp of ``k``
      rather than of ``D``, so ``2 eps k_max L`` over the run, as the planar ladder's
      operator bound takes it.

    `growth` is the stated assumption on how much the run amplifies a perturbation
    over its ten steps (:func:`bounds.propagation_growth` of the nonlinear rate over
    the run is below 2 in every case here).
    """
    schedule = run_request.stepper.schedule
    levels = math.log2(problem.grids.n_to) + math.log2(max(transverse, 1))
    per_step = 4 * (PRODUCTS + FFT_CONSTANT * levels + depth)
    rounding = step_budget(per_step, schedule.n_steps, growth)
    edges = schedule.edges
    swept = sum(
        np.abs(delta_phi(problem.linear, float(a), float(b)))
        for a, b in itertools.pairwise(edges)
    )
    final = np.abs(delta_phi(problem.linear, float(edges[0]), float(edges[-1])))
    # The exponent keeps the operator tables' leading axis of one.
    argument = (
        EPS * ARGUMENT_FACTOR * np.broadcast_to(swept + final, (1, *problem.shape))[0]
    )
    bins = populated([problem.initial.field0], 1e-3)
    separable = problem.linear.separable
    roots = (
        0.0
        if separable is None
        else 2
        * EPS
        * math.sqrt(float(np.max(np.abs(separable.k_squared[separable.active]))))
        * float(schedule.z_end - schedule.z_start)
    )
    return rounding + growth * (float(np.max(argument[bins])) + roots)


def replay(
    backend_class, problem, run_request, grid, *, modes=None, noise=None, **options
):
    """Replay the problem's schedule through its planes; return the final state.

    The backend's own session runs it (SEP 47 phase 5), from the problem's entrance.
    """
    backend = backend_class(problem, run_request, grid=grid, modes=modes, **options)
    try:
        start = Start.entrance(problem, noise=noise)
        field = run(backend, problem, run_request, start).fields[-1]
        assert np.all(np.isfinite(field))
        return field
    finally:
        backend.close()


def assert_compiled(
    backend_class, problem, run_request, grid, limit, *, what="march", **replaying
):
    """Replay on the reference and `backend_class`; assert it within `limit`.

    Returns the departure measured.
    """
    expected = replay(NumpyRefBackend, problem, run_request, grid, **replaying)
    got = replay(backend_class, problem, run_request, grid, **replaying)
    assert_within(
        got, expected, limit, what=f"{backend_class.name} against numpy_ref, {what}"
    )
    return departure(got, expected)


def assert_parity(
    backend_class,
    problem,
    run_request,
    grid,
    *,
    modes=None,
    noise=None,
    options=None,
    **bound,
):
    """Replay on both backends and assert the states within :func:`parity_bound`.

    `options` are both backends' execution options, an exact node chunk among them.
    """
    limit = parity_bound(problem, run_request, **bound)
    return (
        assert_compiled(
            backend_class,
            problem,
            run_request,
            grid,
            limit,
            modes=modes,
            noise=noise,
            options=options,
        ),
        limit,
    )


def uniform(length, steps=STEPS, **planes):
    """Return a uniform schedule of `steps` steps over `length`."""
    return FixedSchedule.uniform(0.0, length, length / steps * 1.0000001, **planes)


def with_mode(problem, run_request, mode):
    """Return the problem and its request with the propagator mode set."""
    return Translation(
        problem,
        run_request.with_stepper(
            dataclasses.replace(run_request.stepper, propagator_mode=mode)
        ),
    )


def varying_capillary():
    """Return a nitrogen gradient capillary with Raman and plasma under the march.

    A varying operator: the tabulated flows are the step's ``P`` and ``Q``, and the
    recomputed ones are formed in turn in one held scratch (SEP 24 section 4.3).
    """
    length = 0.02
    return Simulation.from_capillary(
        100e-6,
        length,
        "N2",
        (1.0, 0.7),
        wavelength=800e-9,
        fwhm=10e-15,
        energy=20e-6,
        trange=150e-15,
        wavelength_limits=(200e-9, 2e-6),
        raman=True,
        plasma=True,
        noise=None,
        n_save=2,
        schedule=uniform(length),
        dz_max=length / STEPS * 1.0000001,
        dz_init=length / STEPS,
        integrator=MARCH,
        backend="numpy_ref",
        # The in-stage collar would hold a step to 0.44 mm; the split-step one
        # exercises the accepted step's own treatment instead.
        absorber_scheme="split_step",
    )

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage march's parity against the reference: SEP 24 7.2's bound and runs.

Each parity case replays one schedule of at most ten steps on the reference and on a
compiled backend, and the states agree within a rounding bound fixed before the
comparison (SEP 25; :func:`parity_bound`): no agreement is bitwise. The cases themselves
are ``test_native_low_storage.py``'s.
"""

import dataclasses
import itertools
import math

import numpy as np

from backends.test_low_storage import ARGUMENT_FACTOR, MARCH, populated
from bounds import EPS, assert_within, departure, step_budget
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.stepper import FixedSchedule
from solprop.translate.linop import delta_phi

STEPS = 10
"""The qualification scope's propagation (AGENTS.md): ten steps."""

PRODUCTS = 32
"""Roundings of one evaluation outside its transforms, with a margin: the input's add
and scale, the prefactor, the window and the absorber's term, the fold's four products
and adds, the flow's exponential (a few eps), and the response's own few per sample."""

FFT_CONSTANT = 8
"""Roundings per level of a transform pair, Higham's ``log2 n`` with a margin."""


def parity_bound(problem, *, growth=2.0, transverse=1, depth=0):
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
    schedule = problem.stepper.schedule
    levels = math.log2(problem.grids.n_to) + math.log2(max(transverse, 1))
    per_step = 4 * (PRODUCTS + FFT_CONSTANT * levels + depth)
    rounding = step_budget(per_step, schedule.n_steps, growth)
    edges = schedule.edges
    swept = sum(
        np.abs(delta_phi(problem.linear, float(a), float(b)))
        for a, b in itertools.pairwise(edges)
    )
    final = np.abs(delta_phi(problem.linear, float(edges[0]), float(edges[-1])))
    argument = EPS * ARGUMENT_FACTOR * np.broadcast_to(swept + final, problem.shape)
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


def replay(backend_class, problem, grid, *, modes=None, noise=None, **options):
    """Replay the problem's schedule through its planes; return the final state."""
    backend = backend_class(problem, grid=grid, modes=modes, **options)
    try:
        backend.set_initial(problem.initial.field0, noise)
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
        field = backend.observe().require_field().copy()
        assert np.all(np.isfinite(field))
        return field
    finally:
        backend.close()


def assert_compiled(backend_class, problem, grid, limit, *, what="march", **replaying):
    """Replay on the reference and `backend_class`; assert it within `limit`.

    Returns the departure measured.
    """
    expected = replay(NumpyRefBackend, problem, grid, **replaying)
    got = replay(backend_class, problem, grid, **replaying)
    assert_within(
        got, expected, limit, what=f"{backend_class.name} against numpy_ref, {what}"
    )
    return departure(got, expected)


def assert_parity(
    backend_class, problem, grid, *, modes=None, noise=None, options=None, **bound
):
    """Replay on both backends and assert the states within :func:`parity_bound`.

    `options` are both backends' execution options, an exact node chunk among them.
    """
    limit = parity_bound(problem, **bound)
    return (
        assert_compiled(
            backend_class,
            problem,
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


def with_mode(problem, mode):
    """Return the problem with its propagator mode set."""
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(problem.stepper, propagator_mode=mode),
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

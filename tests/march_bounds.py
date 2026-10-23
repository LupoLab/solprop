# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Budgets for two runs of one discrete march, and envelopes for adaptive ones.

The driver, output and scan tests compare whole runs: a batch member with its solo run,
a scan point with the same simulation alone, a resumed file with the uninterrupted one,
two entry points that build one problem, a run with a stored record of itself. SEP 25
classes each pair (``tests/bounds.py``):

- **Matched marches** (class R) take the same method, orders, startup and history on a
  prescribed schedule, so they differ by rounding alone. :func:`march_bound` is their
  relative normwise budget, fixed from the operation counts below before any run.
  :data:`FIXED_ORDER` is the Adams method these tests pin when the default's order
  choice would be the one decision that rounding could fork on a fixed schedule.
- **Adaptive pairs** (class T) may take different steps; each is compared with a run
  of the same problem at a hundredth of its tolerance, at an envelope set from the
  uninterrupted run's own departure from it (:func:`assert_in_envelope`).

The counts, per step and relative to the state, both runs' roundings added:

- the linear half, ``32 u``: the ladder's 16 eps per step (``docs/dev/accuracy.md``
  section 7), a phase factor and a mask per bin;
- each source evaluation, ``2 u (M + 2 eta ceil(log2 n_to) + 30 + 3 Q + 2)``: the
  FFT round trip with Higham's ``eta = 7``, the modal synthesis and projection over
  ``Q`` nodes and ``M`` channels, the tensor contraction's 30 roundings
  (``tests/general_parity.py``'s source bound with ``kappa = 1``); a Raman response
  adds its convolution's round trip on the doubled axis, and a plasma its running
  sums, ``4 n_to eps`` (``docs/dev/accuracy.md`` section 7);
- that source reaches the step multiplied by ``dz`` and the step's quadrature weights,
  so relative to the state it is scaled by ``(Phi / S) W``, with ``Phi`` the launch's
  nonlinear phase ``L ||N(a)|| / ||a||``, ``S`` the steps and ``W``
  :func:`bounds.step_evaluations` (the Adams weights' absolute sum, or DOPRI5's stage
  count);
- ``S`` steps add, and the nonlinear half carries an early departure by at most
  ``exp(Phi)`` (SEP 25 section 4.3, :func:`bounds.propagation_growth`).

The cases here are weakly nonlinear (``Phi`` of order one at most, declared per call
where it is larger) and their linear operators are unitary or damping. :data:`MARGIN`
is the safety factor, fixed here before the comparisons were run.
"""

import math

import numpy as np

from bounds import (
    EPS,
    UNIT_ROUNDOFF,
    AccuracyEnvelope,
    assert_within,
    fft_bound,
    propagation_growth,
    step_evaluations,
)
from solprop.stepper import Multistep

FIXED_ORDER = Multistep(variable_order=False)
"""The default Adams method at its top order: no order choice for rounding to fork."""

LINEAR_STEP = 32.0
"""Both runs' rounding of one step's linear half, in units of ``u``."""

FFT_CONSTANT = 7.0
"""``eta / u`` of Higham's Cooley-Tukey bound with accurately computed twiddles."""

CONTRACTION = 30.0
"""Roundings of one sample's nonlinear contraction (``tests/general_parity.py``)."""

MARGIN = 4.0
"""The safety factor on every budget of this module."""

TIGHTER = 100.0
"""How much tighter than the run's own tolerance its reference run is."""

ENVELOPE_FACTOR = 4.0
"""The envelope's safety factor on the uninterrupted run's error."""


def evaluation_rounding(
    n_to: int,
    n_nodes: int = 1,
    n_modes: int = 1,
    *,
    raman: bool = False,
    plasma: bool = False,
) -> float:
    """Return both runs' rounding of one source evaluation, relative to the source."""
    count = (
        n_modes
        + 2 * FFT_CONSTANT * math.ceil(math.log2(n_to))
        + CONTRACTION
        + 3 * n_nodes
        + 2
    )
    if raman:
        count += 2 * FFT_CONSTANT * math.ceil(math.log2(2 * n_to)) + CONTRACTION
    rounding = 2 * UNIT_ROUNDOFF * count
    if plasma:
        rounding += 2 * 4 * n_to * EPS
    return rounding


def march_bound(
    problem,
    steps: int | None = None,
    *,
    phi: float = 1.0,
    kappa: float = 1.0,
    margin: float = MARGIN,
    backend: str = "numpy_ref",
) -> float:
    """Return the relative normwise budget of two matched runs of `problem`.

    Parameters
    ----------
    problem : KernelProblem
        The translated problem; its grid, transform and integrator set the counts.
    steps : int, optional
        Steps taken; the prescribed schedule's by default.
    phi : float, optional
        The declared upper bound on the run's nonlinear phase ``Phi``.
    kappa : float, optional
        The conditioning of the general route's projected sum
        (``general_parity.conditioning``), which scales its source's rounding; one
        on the fixed quadratures.
    margin : float, optional
        The safety factor.
    backend : str, optional
        A device build of Kokkos is held to its predeclared FFT envelope instead,
        ``fft_bound`` per evaluation, when that is larger.

    Returns
    -------
    float
        ``margin exp(Phi) S (32 u + W (Phi / S) eps_eval)``.
    """
    if steps is None:
        schedule = problem.stepper.schedule
        assert schedule is not None, "a matched march needs a prescribed schedule"
        steps = int(schedule.n_steps)
    n_to = int(problem.grids.n_to)
    evaluations = step_evaluations(problem)
    per_evaluation = kappa * evaluation_rounding(
        n_to,
        int(problem.transform.n_nodes),
        int(problem.shape[1]),
        raman=bool(problem.flags.has_raman),
        plasma=bool(problem.flags.has_plasma),
    )
    if backend == "native":
        from bounds import device_backend  # noqa: PLC0415

        if device_backend(backend) != "none":
            per_evaluation = max(per_evaluation, fft_bound(n_to))
    per_step = (
        LINEAR_STEP * UNIT_ROUNDOFF + evaluations * (phi / steps) * per_evaluation
    )
    return margin * propagation_growth(1.0, phi) * steps * per_step


def measured_scales(simulation) -> tuple[float, float]:
    """Return a run's launch ``Phi`` and its source's conditioning ``kappa``.

    Read off the problem and its launch by the reference backend, as
    ``tests/flow_parity.py`` reads them, before anything is compared: ``Phi = L
    ||N(a_0)|| / ||a_0||`` over the run's length, and ``kappa`` the general route's
    projected-sum conditioning at the launch (one on a fixed quadrature).
    """
    from general_parity import conditioning  # noqa: PLC0415
    from solprop.backends.numpy_ref import NumpyRefBackend  # noqa: PLC0415
    from solprop.backends.numpy_ref.backend import _general_maps  # noqa: PLC0415

    problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        state = problem.initial.field0
        bundle = backend._stage_scalars(0.0, 0.0)
        source = backend._nonlinear(state, bundle, np.ones_like(state))
    finally:
        backend.close()
    length = float(problem.stepper.save_planes[-1])
    phi = length * float(np.linalg.norm(source) / np.linalg.norm(state))
    maps = _general_maps(problem)
    kappa = 1.0
    if maps is not None:
        kappa = conditioning(
            maps.at(bundle.transform_weights), state[0], simulation.grid
        )
    return phi, kappa


def simulation_bound(simulation, *, margin: float = MARGIN, backend="numpy_ref"):
    """Return :func:`march_bound` of a simulation at its measured ``Phi`` and ``kappa``.

    ``Phi`` is taken no smaller than one tenth, so that a launch whose source is weak
    at the entrance and grows along the run is still budgeted for.
    """
    phi, kappa = measured_scales(simulation)
    return march_bound(
        simulation.translate(),
        phi=max(phi, 0.1),
        kappa=kappa,
        margin=margin,
        backend=backend,
    )


def assert_matched(actual, expected, bound: float, *, what: str = "") -> float:
    """Assert two matched runs' arrays agree normwise inside `bound`; return the margin.

    A stack of planes is compared plane by plane, each relative to its own norm, so an
    attenuated late plane is not judged against the entrance's scale; a plane that is
    identically zero must stay so.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.ndim <= 1 or actual.shape != expected.shape:
        return assert_within(actual, expected, bound, what=what)
    margins = [0.0]
    for index, (one, other) in enumerate(zip(actual, expected, strict=True)):
        if not np.any(other):
            # A plane that is identically zero by construction (the entrance's noise
            # phase, a channel never launched) stays exactly zero: structural.
            assert not np.any(one), f"{what} plane {index}: not the exact zero"
            continue
        margins.append(assert_within(one, other, bound, what=f"{what} plane {index}"))
    return max(margins)


def assert_in_envelope(runs, reference, uninterrupted, *, what: str = "") -> None:
    """Assert adaptive runs meet the uninterrupted run's accuracy envelope.

    `reference` is the same problem at a :data:`TIGHTER` tolerance. The envelope is
    :data:`ENVELOPE_FACTOR` times the uninterrupted run's own departure from it,
    taken before any of `runs` is compared, and every one of `runs` must meet it
    (SEP 25 section 4.4).
    """
    reference = np.asarray(reference)
    error = float(
        np.linalg.norm(np.asarray(uninterrupted) - reference)
        / np.linalg.norm(reference)
    )
    assert error > 0.0, "the reference run is the run itself"
    envelope = AccuracyEnvelope(
        reference=f"{what} at a {TIGHTER:g} times tighter tolerance",
        error=error,
        factor=ENVELOPE_FACTOR,
    )
    for index, run in enumerate(runs):
        assert_within(run, reference, envelope.bound, what=f"{what} run {index}")


STATISTIC_FACTOR = 4.0
"""How many times the field's relative departure a stored statistic may move.

Twice for a statistic quadratic in the field (energy, power, a spectrum), and one more
factor of two for a ratio of two of them or a width interpolated between samples.
"""


FIELDS = ("field", "fields/values")
"""Stored names that hold the field itself, held to the march's own budget."""


def assert_stored_matched(
    expected, got, bound: float, *, exact=(), factors=None, what=""
):
    """Assert two matched runs' stored datasets agree, name by name.

    `expected` and `got` map dataset names to arrays. Integer, boolean and string data
    are accounting and identity, compared exactly, as is every name in `exact` (the
    positions and steps a schedule prescribes). Floating-point data keep their NaN
    positions exactly (an undefined statistic is undefined in both) and agree within
    `bound` for the field, :data:`STATISTIC_FACTOR` times it for anything else, or
    the factor `factors` names for it (a rate statistic's sensitivity to the field).
    """
    factors = {} if factors is None else factors
    assert set(expected) == set(got)
    for name in expected:
        reference, other = np.asarray(expected[name]), np.asarray(got[name])
        label = f"{what} {name}".strip()
        if name in exact or reference.dtype.kind not in "fc":
            assert np.array_equal(reference, other), label
            continue
        undefined = np.isnan(reference)
        assert np.array_equal(undefined, np.isnan(other)), label
        reference = np.where(undefined, 0.0, reference)
        other = np.where(undefined, 0.0, other)
        if not np.any(reference):
            assert not np.any(other), f"{label}: not the exact zero"
            continue
        scale = 1.0 if name.endswith(FIELDS) else STATISTIC_FACTOR
        assert_matched(other, reference, factors.get(name, scale) * bound, what=label)


PRESCRIBED = (
    "z",
    "step_history/z",
    "step_history/h",
    "step_history/accepted",
    "step_stats/z",
)
"""What a prescribed schedule fixes in a stored run, whatever the arithmetic: exact."""


def assert_points_matched(
    result, reference, scan, builder, *, factors=None, what="point"
):
    """Assert every point of a scan agrees with the same point run another way.

    Both scans step each point on its prescribed schedule, so a point is a matched
    march (a batch member against its solo run, a pool worker against the local loop):
    its stored arrays are held to :func:`simulation_bound` of the point's own
    simulation by :func:`assert_stored_matched`, the schedule's positions exactly,
    and the point's recorded configuration exactly.
    """
    for point in scan.points():
        actual, expected = result.point(point.index), reference.point(point.index)
        assert actual.keys() == expected.keys()
        assert (
            actual["metadata"]["configuration"] == expected["metadata"]["configuration"]
        )
        assert_stored_matched(
            {name: value for name, value in expected.items() if name != "metadata"},
            {name: value for name, value in actual.items() if name != "metadata"},
            simulation_bound(builder(point)),
            exact=PRESCRIBED,
            factors=factors,
            what=f"{what} {point.index}",
        )

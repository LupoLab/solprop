# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's plasma statistics, the envelope's harmonic and a one-tap rule (SEP 43).

From the retired ``test_numba_cpu.py``, ``test_numba_cpu_kernels.py`` and
``test_numba_band_kernels.py``: native against ``numpy_ref``, on their inputs.
"""

import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("solprop_native")


from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

from backends.matched import stash_bound
from backends.test_band_plasma_parity import case as band_case
from backends.test_band_plasma_parity import options_for
from bounds import assert_within
from sessions.runs import launch_statistics, run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata.gases import Gas
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import DOPRI5, FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import assemble

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning",
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning",
)

UNIT = np.finfo(float).eps / 2


def ionising_capillary(field_grid):
    """Return the retired test's gradient capillary of argon, ionising, two steps."""
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    table = build_rate_table(OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov"), (1e9, 2e11))
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=200e-6).build(
        field_grid
    )
    problem, run_request = assemble(
        field_grid,
        mode,
        [
            KerrResponse(chi3=2.2e-52, thg=True),
            PlasmaResponse(Gas.Ar, table, preionisation=1e-3),
        ],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[2e-3],
        schedule=FixedSchedule.uniform(0.0, 2e-3, 1e-3),
        dz_max=0.01,
        dz_init=1e-4,
    )
    return problem, run_request, [mode]


@pytest.mark.medium
def test_every_statistic_matches_the_reference(field_grid):
    """``numba_cpu``'s test on its input and its bound, formed before the comparison.

    This run is strongly ionising and ill-conditioned: a launch moved by one ulp in
    random directions ends far away in the field after its two steps. Two backends
    differ by rounding at every stage, and each such seed is carried the same way, so
    each statistic is bounded by its own measured response to a one-ulp launch
    perturbation times the count of seeds, two steps of seven stages, each at most a
    transform's ``7 log2(n_to)`` ulp, with ``gamma(n_to)`` for the evaluation itself, a
    factor of four for sampling and the retired test's floor of ``4 x 5.4e-13``. The
    response is the reference's own, so it does not see native at all.
    """
    problem, run_request, modes = ionising_capillary(field_grid)
    names = (
        "energy",
        "peak_power",
        "electron_density",
        "peak_ionisation_rate",
        "peak_intensity",
    )
    spec = StatsSpec(names=(*names, "quadrature_error"))
    request = run_request.for_run(stats=spec)

    def statistics(backend_class, launch):
        # Each backend's own session, from `launch` to the plane at 2 mm.
        backend = backend_class(problem, request, grid=field_grid, modes=modes)
        try:
            start = Start(field=np.array(launch), z=float(problem.initial.z))
            return run(backend, problem, request, start).all_planes()[-1].stats
        finally:
            backend.close()

    launch = np.asarray(problem.initial.field0)
    ours = statistics(NativeBackend, launch)
    theirs = statistics(NumpyRefBackend, launch)
    assert set(ours) == set(theirs)
    generator = np.random.default_rng(7)
    response = dict.fromkeys(names, 0.0)
    for _ in range(3):
        signs = generator.choice([-1.0, 1.0], size=launch.shape)
        moved = statistics(NumpyRefBackend, launch * (1 + UNIT * signs))
        for name in names:
            change = np.max(np.abs(moved[name] / theirs[name] - 1))
            response[name] = max(response[name], float(change))
    seeds = 2 * DOPRI5.n_stages * 7 * math.log2(field_grid.n_to)
    scan = 2 * field_grid.n_to * UNIT / (1 - 2 * field_grid.n_to * UNIT)
    for name in names:
        assert np.all(np.asarray(ours[name]) > 0.0)
        assert ours[name] == pytest.approx(
            theirs[name], rel=max(4.0 * seeds * response[name] + scan, 4 * 5.4e-13)
        )
    # A mode-averaged run's rule has no embedded pair, so the difference of the two
    # projections is identically zero; both report that as nothing.
    assert math.isnan(float(ours["quadrature_error"]))
    assert math.isnan(float(theirs["quadrature_error"]))


def test_the_trapezoid_statistics_are_the_reference_scan(field_grid):
    """``numba_cpu``'s ``test_the_statistics_column_is_the_reference_scan``, trapezoid.

    The launch's own plasma statistics, on the reference's fused rule, against the
    reference's scan of the same state. The retired test read a column at ``4 n eps``
    and one ulp; native's column is read through its statistic, so the two synthesise
    the one state and are held to :func:`matched.stash_bound`, which was derived for
    that.
    """
    problem, run_request, modes = ionising_capillary(field_grid)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, plasma_integral=RunningIntegral.trapezoid()
        ),
    )
    spec = StatsSpec(names=("electron_density", "peak_ionisation_rate"))
    request = run_request.for_run(stats=spec)
    values = {}
    for backend_class in (NumpyRefBackend, NativeBackend):
        backend = backend_class(problem, request, grid=field_grid, modes=modes)
        try:
            values[backend_class] = launch_statistics(backend, problem, request)
        finally:
            backend.close()
    theirs, ours = values[NumpyRefBackend], values[NativeBackend]
    rate = OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov")
    peak = float(np.max(theirs["peak_ionisation_rate"]))
    assert peak > 0
    bound = stash_bound(rate, peak, (1e9, 2e11), field_grid.n_to)
    for name in spec.names:
        assert_within(ours[name], theirs[name], bound, what=name)


@pytest.mark.parametrize("name", ["mode_averaged", "radial"])
def test_a_band_rule_of_one_tap_is_the_reference(name):
    """``K = 0`` against the reference directly, ``numba_cpu``'s band column's other
    rule.

    ``native/tests/test_band_plasma.py``'s test of this rule compares two native
    columns, and is marked for a device only. The retired ``64 n_to eps`` for the one
    tap.
    """
    from backends.test_radial_nonlinear import derivative  # noqa: PLC0415

    grid, problem, run_request = band_case(name)
    flat = RunningIntegral(
        "band", np.ones(1), dt=grid.dt_fine, band_edge=grid.omega_max_window
    )
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(problem.responses, plasma_integral=flat),
    )
    bound = 64 * grid.n_to * np.finfo(float).eps
    options = options_for(name)
    expected = derivative(NumpyRefBackend, problem, run_request, grid, options)
    actual = derivative(NativeBackend, problem, run_request, grid, options)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0
    assert float(np.max(np.abs(actual - expected))) <= bound * scale


def test_the_envelope_harmonic_column_is_the_reference(thg_problem, thg_grid):
    """``numba_cpu``'s ``test_the_envelope_column...``, ``thg=True``, one evaluation.

    The envelope's harmonic term has had no parity on native: the surviving test only
    sees that the term does something, and another holds it to the field grid. The
    retired bound was the column's ``8 eps``, without the transforms. Native's column is
    read through the first stage, so the free ladder's evaluation budget, ``32 eps`` per
    FFT level, and the term has to move the answer by more than that.
    """
    from backends.test_radial_nonlinear import derivative  # noqa: PLC0415

    def noisy(problem):
        """Return `problem` launching a seeded field on every active bin.

        A band-limited soliton leaves the harmonic far below this budget, so the launch
        fills the band, as the retired test's random column did.
        """
        generator = np.random.default_rng(11)
        launch = np.zeros_like(problem.initial.field0)
        active = problem.grids.active_mask
        count = int(np.count_nonzero(active))
        launch[..., active] = (
            generator.standard_normal(count) + 1j * generator.standard_normal(count)
        ) * float(np.max(np.abs(problem.initial.field0)))
        return dataclasses.replace(
            problem, initial=dataclasses.replace(problem.initial, field0=launch)
        )

    harmonic, run_request = thg_problem(thg=True)
    problem = noisy(harmonic)
    expected = derivative(NumpyRefBackend, problem, run_request, thg_grid)
    actual = derivative(NativeBackend, problem, run_request, thg_grid)
    scale = float(np.max(np.abs(expected)))
    assert scale > 0
    bound = 32 * np.finfo(float).eps * math.ceil(math.log2(thg_grid.n_to))
    assert float(np.max(np.abs(actual - expected))) <= bound * scale
    plain, plain_request = thg_problem(thg=False)
    without = derivative(NumpyRefBackend, noisy(plain), plain_request, thg_grid)
    assert float(np.max(np.abs(without - expected))) > bound * scale

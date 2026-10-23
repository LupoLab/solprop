# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""P1 of the verification ladder: the N-soliton recurrence, with its phase.

``docs/dev/stepper.md`` section 11, rung P1, and trap 12: check the recurrence **with
its phase**, because an amplitude-only metric understates the error by one to two
orders. The global phase after one soliton period is ``+pi/4`` in the optics convention,
so ``-pi/4`` in the internal one, and it is asserted rather than fitted.

The problem is the one ``tests/driver/test_scaffold.py`` propagates with fixed-step
IP-RK4, so the two integrators can be compared directly: the scaffold is an independent
check of the stage passes rather than of the physics they share.
"""

import math

import numpy as np
import pytest

from convergence_records import Column, Study, record
from solprop.driver.scaffold import propagate_fixed_rk4
from solprop.stepper import FixedSchedule, PIController, observed_order

# Slow and benchmark ladders use the environment-selected fixture, which defaults to
# Numba and is set to native for the native suite's lane.


def propagate(backend_class, problem, grid, distance):
    """Run a problem to a distance and return the masked input and the final state."""
    backend = backend_class(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(distance)
    final = backend.observe().require_field()[0, 0, 0]
    backend.close()
    start = problem.initial.field0[0, 0, 0] * problem.grids.active_mask
    return summary, start, final


def recurrence_error(final, start, phase):
    """Return the phase-resolved error as a fraction of the input's norm."""
    return float(
        np.linalg.norm(final - np.exp(1j * phase) * start) / np.linalg.norm(start)
    )


def realised_phase(final, start):
    """Return the global phase actually realised, read at the brightest bin."""
    peak = int(np.argmax(np.abs(start)))
    return float(np.angle(final[peak] / start[peak]))


# The two records of `docs/dev/accuracy.md` section 4. Both ladders run on a fixed
# schedule, so the trajectory is deterministic and only round-off separates one machine
# from another -- but both end at a floor where round-off is the whole of what is left,
# and that is what `atol` carries. `rtol` is the width of each test's own claim.
SOLITON_N3 = Study(
    name="soliton_n3",
    title="The N = 3 soliton's recurrence against the step count",
    case={
        "problem": "N = 3 soliton of the GNLSE fixture, one soliton period",
        "grid": "the narrow soliton grid, 4096 bins over 4 ps",
        "metric": "phase-resolved recurrence error, ||final - e^{i phi} start|| / "
        "||start||, with phi = -pi/4 (the internal convention)",
        "schedule": "uniform fixed steps",
    },
    axis={"name": "n_steps", "unit": "steps", "label": "uniform steps per period"},
    columns=(
        Column(
            name="error",
            unit="1 (relative to the input norm)",
            why="a fixed schedule is a deterministic trajectory, so above the floor "
            "only round-off separates two machines; atol is a hundred times the "
            "round-off of the recurrence itself, which no rung of this ladder reaches",
            rtol=1e-3,
            atol=1e-12,
        ),
        Column(
            name="order",
            unit="1",
            why="a log2 of two errors each known to 1e-3, so far tighter than the "
            "4.5 to 8.0 the test asserts; null on the coarsest rung, which has no pair",
            rtol=0.0,
            atol=0.05,
        ),
    ),
)

SOLITON_N6 = Study(
    name="soliton_n6",
    title="Milestone M1: the N = 6 soliton's recurrence on a band that holds it",
    case={
        "problem": "N = 6 soliton of the GNLSE fixture, one soliton period",
        "grid": "the wide grid, 350 nm to 3 um over 4 ps, 8192 bins",
        "metric": "phase-resolved recurrence error, as above",
        "schedule": "uniform fixed steps",
        "backend": "the default backend only (docs/dev/testing.md section 2)",
    },
    axis={"name": "n_steps", "unit": "steps", "label": "uniform steps per period"},
    columns=(
        Column(
            name="error",
            unit="1 (relative to the input norm)",
            why="a fifth: the ladder ends on the band's own floor of about 5e-9, "
            "where sixty-four thousand steps of a different transform accumulate "
            "differently, and atol carries that floor outright",
            rtol=0.2,
            atol=1e-9,
        ),
        Column(
            name="order",
            unit="1",
            why="what a fifth in one rung costs a log2 ratio; null on the coarsest "
            "rung, which has no pair, and meaningless once a rung is on the floor",
            rtol=0.0,
            atol=0.5,
        ),
    ),
)


def wrapped(phase):
    """Return a phase folded into ``(-pi, pi]``, as :func:`numpy.angle` returns one."""
    return float(np.angle(np.exp(1j * phase)))


def test_the_fundamental_soliton_is_stationary(
    backend_class, soliton_problem, soliton_grid, scales
):
    """Ten dispersion lengths, and only the phase ``-z/2L_D`` changes."""
    distance = 10 * scales.dispersion_length
    problem = soliton_problem(
        soliton_grid,
        1,
        save_planes=[distance],
        controller=PIController(rtol=1e-10, atol=1e-18),
        dz_max=scales.dispersion_length,
        dz_init=1e-4,
    )
    summary, start, final = propagate(backend_class, problem, soliton_grid, distance)
    # Measured 1.3e-11 over 929 steps; the bound is 1e-9. What is left is the
    # local error the controller was asked for, accumulated over the run, not the grid.
    assert recurrence_error(final, start, -5.0) < 1e-9
    assert realised_phase(final, start) == pytest.approx(wrapped(-5.0), abs=1e-9)
    assert summary.met_tolerance


def n6_rows(series, counts, errors):
    """Return the record rows of one N = 6 ladder."""
    return [
        {
            "series": series,
            "axis": count,
            "error": float(error),
            "order": None
            if index == 0
            else float(observed_order(errors[index - 1], error)),
        }
        for index, (count, error) in enumerate(zip(counts, errors, strict=True))
    ]


@pytest.mark.slow
def test_the_third_order_soliton_recurs_at_the_scheme_s_order(
    soliton_problem, soliton_grid, scales, request, selected_backend_class
):
    """Three levels and an order gate, because a two-level extrapolant cannot fire."""
    errors = []
    for n_steps in (200, 400, 800):
        problem = soliton_problem(
            soliton_grid,
            3,
            save_planes=[scales.period],
            dz_max=scales.period,
            dz_init=scales.period / n_steps,
            schedule=FixedSchedule.uniform(0.0, scales.period, scales.period / n_steps),
        )
        _, start, final = propagate(
            selected_backend_class, problem, soliton_grid, scales.period
        )
        errors.append(recurrence_error(final, start, scales.recurrence_phase))
    orders = [
        observed_order(errors[index], errors[index + 1])
        for index in range(len(errors) - 1)
    ]
    # Measured 3.72e-2, 1.09e-3 and 1.92e-5 at 200, 400 and 800 steps, so orders 5.10
    # and 5.82: fifth order, with the coarsest rung not yet fully asymptotic, which is
    # why the second reading is the higher one. The gate of testing.md section 1
    # rejects anything above p + 3 = 8.
    assert errors[-1] < 1e-4
    assert all(4.5 < order < 8.0 for order in orders), orders
    record(
        SOLITON_N3,
        request=request,
        backend=selected_backend_class.name,
        rows=[
            {
                "series": "N = 3, fixed schedule",
                "axis": count,
                "error": float(error),
                "order": None if index == 0 else float(orders[index - 1]),
            }
            for index, (count, error) in enumerate(
                zip((200, 400, 800), errors, strict=True)
            )
        ],
    )


@pytest.mark.slow
def test_the_third_order_soliton_recurs_with_its_phase(
    soliton_problem, soliton_grid, scales, selected_backend_class
):
    """Adaptively, at a tolerance that gets the recurrence to ten digits."""
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[scales.period],
        controller=PIController(rtol=1e-10, atol=1e-18),
        dz_max=scales.period / 20,
        dz_init=1e-4,
    )
    _, start, final = propagate(
        selected_backend_class, problem, soliton_grid, scales.period
    )
    # Measured 1.70e-10 over 2755 steps, and the phase to nine decimals.
    assert recurrence_error(final, start, scales.recurrence_phase) < 1e-9
    assert realised_phase(final, start) == pytest.approx(
        scales.recurrence_phase, abs=1e-8
    )


# The scaffold's step count for the agreement below, and the backend's tolerance beside
# it: a few hundred steps each, which is what makes this a medium test rather than the
# 4000-step slow one it replaced.
AGREEMENT_STEPS = 400
AGREEMENT_RTOL = 1e-6


@pytest.mark.medium
def test_the_reference_backend_and_the_scaffold_agree(
    backend_class, soliton_problem, soliton_grid, gnlse, scales
):
    """An independent integrator on the same problem: the passes, not the physics.

    ``propagate_fixed_rk4`` composes the nonlinear term from the physics layer and steps
    it with the classical interaction-picture RK4, sharing nothing with the backend's
    passes but the response itself, so what the two agreeing tests is the stage
    combination and the propagator factors.

    The claim is asserted against the scaffold's **own** recurrence error, computed
    here rather than quoted, which is what makes a few hundred steps enough: the backend
    at ``rtol = 1e-6`` sits two orders below the fourth-order scaffold at 400 steps, so
    the gap between the two integrators *is* the scaffold's distance from the analytic
    answer and nothing else.
    """
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[scales.period],
        controller=PIController(rtol=AGREEMENT_RTOL, atol=1e-18),
        dz_max=scales.period / 20,
        dz_init=1e-4,
    )
    _, start, final = propagate(backend_class, problem, soliton_grid, scales.period)
    scaffold = propagate_fixed_rk4(
        gnlse, soliton_grid, start, scales.period, AGREEMENT_STEPS
    )[1][-1]
    gap = float(np.linalg.norm(final - scaffold) / np.linalg.norm(start))
    scaffold_error = recurrence_error(scaffold, start, scales.recurrence_phase)
    # Measured: the scaffold is 4.930e-4 from the analytic answer at 400 steps and the
    # backend 2.908e-6 from it over 470 steps, none rejected, so the gap reads 4.922e-4
    # and equals the scaffold's own error to 1.7e-3 relative. The 5 % asserted is thirty
    # times that, and asserting against a computed quantity rather than a constant is
    # what stops a backend returning zeros -- or returning the input unchanged -- from
    # passing.
    assert scaffold_error > 1e-5
    assert gap == pytest.approx(scaffold_error, rel=0.05)


@pytest.mark.benchmark
def test_the_sixth_order_soliton_converges_on_the_scaffold_s_grid(
    backend_class, soliton_problem, soliton_grid, scales
):
    """On the scaffold's grid the recurrence converges at the scheme's order once the
    polarisation taper is off, which is the default.

    With the taper on, this grid floors at 2.3e-6 (7.38e-6 at 8000 steps, 2.30e-6 at
    16000, 2.28e-6 at 32000), because the taper removes the polarisation's content in
    the spectral collar, which a compressed N = 6 pulse reaches; the temporal window and
    the span change nothing. Off, the same runs give 7.4e-6 and 2.7e-7: order 4.8
    between the rungs, the coarse one not yet asymptotic.

    Marked ``benchmark``, so **no lane runs it**: at 84 s it was the third of the three
    N = 6 ladders that made the slow lane too slow, it is a study of a taper that is off
    by default, and nothing in the default configuration rests on it. Run it by hand
    with
    ``uv run pytest tests/backends/test_numpy_ref_soliton.py -m benchmark``. The ladder
    the default configuration does rest on is the wide-band one below, which stays
    slow as milestone M1's definition.
    """
    errors = []
    for n_steps in (8000, 16000):
        problem = soliton_problem(
            soliton_grid,
            6,
            save_planes=[scales.period],
            dz_max=scales.period,
            dz_init=scales.period / n_steps,
            schedule=FixedSchedule.uniform(0.0, scales.period, scales.period / n_steps),
        )
        _, start, final = propagate(backend_class, problem, soliton_grid, scales.period)
        errors.append(recurrence_error(final, start, scales.recurrence_phase))
        assert realised_phase(final, start) == pytest.approx(
            scales.recurrence_phase, abs=1e-4
        )
    assert errors[0] < 1e-5
    assert observed_order(errors[0], errors[1]) > 4.0
    assert errors[1] < 5e-7


@pytest.mark.slow
def test_the_sixth_order_soliton_converges_at_the_scheme_s_order_on_a_wide_band(
    soliton_problem, wide_grid, scales, request, selected_backend_class
):
    """Milestone M1: on a band that holds the compressed pulse the ladder is the
    scheme's, down to the band's own floor.

    Measured with the polarisation taper off, the default: 9.2e-6, 6.1e-8, 8.9e-9 and
    4.7e-9 at 8000, 16000, 32000 and 64000 steps, so order 7.2 between the first two
    rungs (the coarse one not yet asymptotic) and a floor of 5e-9 that the band sets.
    With the taper on the ladder is 9.2e-6, 9.6e-8 and 6.1e-8 and floors at 6e-8. The
    two cheapest rungs are run here.
    """
    errors = []
    for n_steps in (8000, 16000):
        problem = soliton_problem(
            wide_grid,
            6,
            save_planes=[scales.period],
            dz_max=scales.period,
            dz_init=scales.period / n_steps,
            schedule=FixedSchedule.uniform(0.0, scales.period, scales.period / n_steps),
        )
        _, start, final = propagate(
            selected_backend_class, problem, wide_grid, scales.period
        )
        errors.append(recurrence_error(final, start, scales.recurrence_phase))
    assert realised_phase(final, start) == pytest.approx(
        scales.recurrence_phase, abs=1e-6
    )
    # Fifth order at least between the two rungs, and the finer one within a factor of
    # two of its measured value.
    assert observed_order(errors[0], errors[1]) > 4.5
    assert errors[1] < 2e-7
    # The two cheap rungs of the record; the two fine ones and the whole taper-on
    # ladder are the benchmark-marked writer below, which no lane runs.
    record(
        SOLITON_N6,
        request=request,
        backend=selected_backend_class.name,
        rows=n6_rows("wide band, taper off", (8000, 16000), errors),
    )


@pytest.mark.benchmark
def test_the_wide_band_ladder_at_every_rung_and_with_the_taper_on(
    soliton_problem, wide_grid, scales, request, selected_backend_class
):
    """The rungs of the M1 ladder that no lane runs, written into the record by hand.

    The slow test above runs the two cheapest rungs, which is what milestone M1 rests
    on; the ladder `docs/dev/accuracy.md` section 4 draws needs the two fine ones as
    well, and the taper-on comparison. Together they are about a hundred and seventy
    thousand steps of a 8192-bin field grid, so this is `benchmark`-marked: no lane runs
    it, and its record is written once.

    Run it, and rewrite its rows of ``tests/data/convergence/soliton_n6.json``, with

        SOLPROP_WRITE_RECORDS=1 uv run pytest \
            tests/backends/test_numpy_ref_soliton.py -m benchmark -p no:randomly

    from a clean tree. Measured with the taper off: 9.2e-6, 6.1e-8, 8.9e-9 and 4.7e-9 at
    8000, 16000, 32000 and 64000 steps -- order 7.2 between the first two, the coarse
    rung not yet asymptotic, and then the band's own floor of about 5e-9. With the taper
    on: 9.2e-6, 9.6e-8 and 6.1e-8, flooring an order higher, because the spectral collar
    takes the wings off the cubic, which is why the taper is off by default.
    """
    for series, taper, counts in (
        ("wide band, taper off", False, (8000, 16000, 32000, 64000)),
        ("wide band, taper on", True, (8000, 16000, 32000)),
    ):
        errors = []
        for n_steps in counts:
            problem = soliton_problem(
                wide_grid,
                6,
                save_planes=[scales.period],
                dz_max=scales.period,
                dz_init=scales.period / n_steps,
                schedule=FixedSchedule.uniform(
                    0.0, scales.period, scales.period / n_steps
                ),
                polarisation_taper=taper,
            )
            _, start, final = propagate(
                selected_backend_class, problem, wide_grid, scales.period
            )
            errors.append(recurrence_error(final, start, scales.recurrence_phase))
        # Each ladder is monotone and each ends on a floor rather than on truncation.
        assert errors == sorted(errors, reverse=True), (series, errors)
        record(
            SOLITON_N6,
            request=request,
            backend=selected_backend_class.name,
            rows=n6_rows(series, counts, errors),
        )


def test_the_recurrence_phase_is_the_conjugate_of_the_optics_convention(scales):
    """The internal FFT convention is the conjugate of the optics one."""
    assert pytest.approx(-math.pi / 4) == scales.recurrence_phase

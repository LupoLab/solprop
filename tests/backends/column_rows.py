# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 43's retired column tests: their rows of the case catalogue.

**What is compared.** The compiled CPU backend's kernel tests (``test_numba_cpu.py``,
``test_numba_cpu_kernels.py`` and ``test_numba_band_kernels.py``), retired with that
backend in SEP 43, held its columns to the reference on inputs of their own. Native took
over their claims against ``numpy_ref`` on the same inputs, and they are these rows:

- the ionising capillary (:func:`ionising_capillary`): argon in a pressure gradient,
  a thousandth pre-ionised and ionising hard, two fixed Dormand-Prince steps;
  ``steps`` compares the field and every statistic of its specification at their end;
- the same capillary under the trapezoid rule: ``launch`` compares the plasma
  statistics the session observes at the launch, nothing stepped;
- a band rule of one tap (``K = 0``), the band column's other rule, on the
  mode-averaged helium capillary and on argon on the radial route (cases of
  ``plasma_column_cases.py``): ``derivative``, one evaluation at the launch;
- the envelope's third harmonic, on a seeded launch that fills the band:
  ``derivative``.

Every field comparison is peak-normalised (``metric="peak"``) but the launch's
statistics, which are held normwise, as their old test held them.

**The bounds are the old tests', fixed before native runs.** The march's statistics are
bounded by their own response to the launch moved by one ulp (:func:`ulp_response`),
measured on the reference, times the count of rounding seeds the march carries
(:func:`held_to_their_response`); the launch's statistics by the stash bound the old
test took from ``matched.py``; the one-tap rule and the harmonic column by the retired
tests' own rounding counts per transform level.

**The guard.** Each row's term is switched off on the translated problem: the plasma
(``optmat_rows.without_plasma``, every population's density zeroed) or the harmonic
(:func:`without_harmonic`). Each guarded rung's reference must then move by more than
``parity_ladder.DETECTION`` (a thousand) of that rung's bounds; every guarded rung
clears it by 4e10 bounds or more (measured). The one-tap rule on the helium capillary
frees no electron at the launch, so its row is kept but declared vacuous.

**How to read a row.** Every row is a :func:`column_row` call, for example::

    column_row(
        name="one-tap band radial",
        route="radial",
        response="plasma",
        features="band rule of one tap",
        why="On argon on the radial route, the band rule of one tap ...",
        build=partial(one_tap, "radial"),
        switch_off=without_plasma,
        rungs={"derivative": one_tap_budget},
        metric="peak",
    )

``name`` makes the test id (``column one-tap band radial``); ``route``, ``response`` and
``features`` are the axes the row covers. ``build`` makes the problem: here the radial
case of ``plasma_column_cases.py`` with its running integral replaced by one tap.
``rungs`` names what is compared, each with its bound: here one evaluation at the
launch, within the retired rounding count, peak-normalised. ``switch_off`` removes the
row's term for the guard. ``why`` is the claim a failure quotes.
"""

import dataclasses
import functools
import math
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from backends.conftest import BETA2, GAMMA, GNLSE_SETTINGS, WAVELENGTH, soliton_field
from backends.matched import stash_bound
from backends.optmat_rows import without_plasma
from backends.plasma_column_cases import case as band_case
from backends.plasma_column_cases import options_for
from parity_ladder import (
    Built,
    Row,
    Statistic,
    peak_departure,
    reference,
    steps,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.gnlse import problem_from
from solprop.nonlinear.ionisation import RunningIntegral
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata.gases import Gas
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import DOPRI5, FixedSchedule, PIController
from solprop.translate.assembly import assemble
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import PropagationProblem

EPS = float(np.finfo(float).eps)
"""The machine epsilon."""

# --- the ionising capillary -----------------------------------------------------------

FIELD_GRID = FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)
"""The field grid of the parity and gradient-capillary tests (``field_grid``)."""

ARGON_RATE = OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov")
"""Argon's tunnelling rate at 800 nm, the capillary's plasma."""

RATE_RANGE = (1e9, 2e11)
"""The field amplitudes the rate is tabulated over, V/m."""

IONISING_STATISTICS = (
    "energy",
    "peak_power",
    "electron_density",
    "peak_ionisation_rate",
    "peak_intensity",
)
"""The ionising march's statistics that measure something; ``quadrature_error`` is
read beside them."""


def ionising_capillary(
    integral: RunningIntegral | None = None, statistics: tuple[str, ...] = ()
) -> Built:
    """Return the retired test's gradient capillary of argon, ionising, two steps.

    A Marcatili capillary of 125 um radius, argon from 0.2 to 2 bar and back, a 10 fs,
    200 uJ pulse at 800 nm, Kerr with its harmonic and argon's plasma a thousandth
    pre-ionised: two fixed steps of a millimetre, saved at 2 mm. `integral` replaces
    the plasma's running integral; `statistics` are read.
    """
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    table = build_rate_table(ARGON_RATE, RATE_RANGE)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=200e-6).build(
        FIELD_GRID
    )
    problem, run_request = assemble(
        FIELD_GRID,
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
    if integral is not None:
        problem = dataclasses.replace(
            problem,
            responses=dataclasses.replace(problem.responses, plasma_integral=integral),
        )
    return Built(
        problem,
        run_request.for_run(stats=StatsSpec(names=statistics)),
        FIELD_GRID,
        modes=(mode,),
    )


# --- the march's bound: its response to one ulp ---------------------------------------

UNIT = EPS / 2
"""The unit roundoff: the launch moves by one in random directions."""

RETIRED_FLOOR = 4 * 5.4e-13
"""The retired test's floor, ``4 x 5.4e-13``."""

RESPONSE = "ulp response"
"""The key the reference's response is kept under in ``Built.readings``."""


def ulp_response(built: Built) -> dict[str, float]:
    """Return the reference's largest relative change under a launch moved by one ulp.

    Per statistic of :data:`IONISING_STATISTICS`, and for the field (``"field"``,
    peak-normalised), over three launches moved by one ulp in random directions, each
    marched as the row is. Measured once on the build, on the reference alone, so that
    it does not see native at all.
    """
    if RESPONSE not in built.readings:
        unmoved = steps(NumpyRefBackend, built, None)
        launch = np.asarray(built.problem.initial.field0)
        generator = np.random.default_rng(7)
        response = dict.fromkeys((*IONISING_STATISTICS, "field"), 0.0)
        for _ in range(3):
            signs = generator.choice([-1.0, 1.0], size=launch.shape)
            moved = steps(
                NumpyRefBackend, launched(built, launch * (1 + UNIT * signs)), None
            )
            for name in IONISING_STATISTICS:
                change = np.max(
                    np.abs(moved.statistics[name] / unmoved.statistics[name] - 1)
                )
                response[name] = max(response[name], float(change))
            response["field"] = max(
                response["field"], peak_departure(moved.field, unmoved.field)
            )
        built.readings[RESPONSE] = response
    return built.readings[RESPONSE]


def launched(built: Built, field: np.ndarray) -> Built:
    """Return the build launching `field` instead of its own launch."""
    problem = built.problem
    return dataclasses.replace(
        built,
        problem=dataclasses.replace(
            problem, initial=dataclasses.replace(problem.initial, field0=field)
        ),
    )


def response_bound(built: Built, name: str) -> float:
    """Bound a quantity of the march by its response to one ulp, times its seeds.

    Two backends differ by rounding at every stage, and each such seed is carried the
    same way as a launch's ulp, so the quantity is bounded by its own measured
    response (:func:`ulp_response`) times the count of seeds: two steps of seven
    stages, each at most a transform's ``7 log2(n_to)`` ulp, with ``gamma(n_to)`` for
    the evaluation itself, a factor of four for sampling and the retired test's floor
    of ``4 x 5.4e-13``.
    """
    n_to = built.grid.n_to
    seeds = 2 * DOPRI5.n_stages * 7 * math.log2(n_to)
    scan = 2 * n_to * UNIT / (1 - 2 * n_to * UNIT)
    return max(4.0 * seeds * ulp_response(built)[name] + scan, RETIRED_FLOOR)


def field_response(built: Built) -> float:
    """Bound the march's field by its response to one ulp (:func:`response_bound`)."""
    return response_bound(built, "field")


def held_to_their_response(built: Built, name: str, field: float) -> Statistic:
    """Return the rule the ionising march holds a statistic to.

    Relative at every element, to the statistic's own :func:`response_bound`, the
    reference's value positive; the quadrature error, which measures nothing on a
    mode-averaged run, must be NaN on both sides.
    """
    if name == "quadrature_error":
        return Statistic(field, nan=True)
    return Statistic(response_bound(built, name), elementwise=True, positive=True)


def launch_stash(built: Built) -> float:
    """Bound the launch's plasma statistics: :func:`matched.stash_bound`.

    At the reference's own peak rate at the launch, which must be positive.
    """
    peak = float(np.max(reference(built, "launch")["peak_ionisation_rate"]))
    assert peak > 0, "the reference's launch does not ionise"
    return stash_bound(ARGON_RATE, peak, RATE_RANGE, built.grid.n_to)


# --- a band rule of one tap -----------------------------------------------------------


def one_tap(name: str) -> Built:
    """Return a case of ``plasma_column_cases.py`` under a band rule of one tap."""
    grid, problem, run_request = band_case(name)
    flat = RunningIntegral(
        "band", np.ones(1), dt=grid.dt_fine, band_edge=grid.omega_max_window
    )
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(problem.responses, plasma_integral=flat),
    )
    return Built(problem, run_request, grid, options=options_for(name))


def one_tap_budget(built: Built) -> float:
    """Bound one evaluation under a one-tap band rule: the retired ``64 n_to eps``."""
    return 64 * built.grid.n_to * EPS


# --- the envelope's harmonic ----------------------------------------------------------

THG_GRID = EnvelopeGrid.from_limits(1.0, WAVELENGTH, (500e-9, 2e-6), 1e-12, thg=True)
"""An envelope grid oversampled six times, which resolves the third harmonic
(``thg_grid``)."""


def harmonic_column() -> Built:
    """Return the GNLSE soliton fibre with its harmonic, launching a seeded field.

    A band-limited soliton leaves the harmonic far below the row's budget, so the
    launch fills the band with a seeded random field of the soliton's peak, as the
    retired test's random column did.
    """
    fibre = problem_from(
        [0.0, 0.0, BETA2], omega_ref=THG_GRID.omega0, gamma=GAMMA, shock=False
    )
    problem, run_request = assemble(
        THG_GRID,
        fibre.mode,
        [KerrResponse(chi3=fibre.kerr.chi3, thg=True)],
        soliton_field(THG_GRID, 1),
        controller=PIController(rtol=1e-9, atol=1e-18),
        save_planes=[1e-3],
        dz_max=1e-3,
        dz_init=1e-5,
        **GNLSE_SETTINGS,
    )
    generator = np.random.default_rng(11)
    launch = np.zeros_like(problem.initial.field0)
    active = problem.grids.active_mask
    count = int(np.count_nonzero(active))
    launch[..., active] = (
        generator.standard_normal(count) + 1j * generator.standard_normal(count)
    ) * float(np.max(np.abs(problem.initial.field0)))
    return launched(Built(problem, run_request, THG_GRID), launch)


def without_harmonic(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem without its harmonic: no harmonic factor, no doubled carrier.

    On an envelope grid those are the two fields ``thg=False`` changes in the
    translation (compared field by field), and the reference's derivative is the old
    test's without the harmonic, to the bit (measured).
    """
    responses = problem.responses
    kerr = responses.kerr_coefficients
    assert kerr is not None
    return dataclasses.replace(
        problem,
        grids=dataclasses.replace(problem.grids, carrier_2w0=None),
        responses=dataclasses.replace(
            responses,
            kerr_coefficients=dataclasses.replace(kerr, harmonic_factor=0.0),
        ),
    )


def transform_levels(built: Built) -> float:
    """Bound one evaluation of the harmonic column: ``32 eps`` per transform level.

    The free ladder's evaluation budget, ``32 eps ceil(log2 n_to)``. The retired bound
    was the column's ``8 eps``, without the transforms; native's column is read
    through the first stage, which takes them.
    """
    return 32 * EPS * math.ceil(math.log2(built.grid.n_to))


# --- the rows -------------------------------------------------------------------------


def column_row(
    *,
    name: str,
    route: str,
    stepper: str = "none",
    response: str,
    features: str,
    build: Callable[[], Built],
    **fields: Any,
) -> Row:
    """Return a row of a retired column test, medium when it marches.

    Its id is ``column {name}``; `fields` are the rest of :class:`parity_ladder.Row`.
    The old module ignored ``MaterialRangeWarning`` and the temporal absorber's
    walk-off warning; no row warns (measured), so none is quieted.
    """
    return Row(
        id=f"column {name}",
        axes={
            "route": route,
            "stepper": stepper,
            "response": response,
            "features": features,
        },
        build=build,
        medium="steps" in fields["rungs"],
        **fields,
    )


HELIUM_FREES_NOTHING_AT_THE_LAUNCH = (
    "The helium launch frees no electron at the launch: switching the plasma off "
    "moves the reference's derivative by 1.1e-4 bounds (measured), rounding. Kept as "
    "the retired test held it, the one-tap rule assembled beside the Kerr term on a "
    "population that frees nothing; the radial row carries the rule's guard."
)
"""Why the one-tap rule on the helium capillary is kept, though vacuous."""


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of the retired column tests."""
    return (
        column_row(
            name="ionising capillary every statistic",
            route="capillary (pressure gradient)",
            stepper="rk",
            response="kerr, plasma",
            features="field grid, ionising, every statistic",
            why="Argon ionising hard in a pressure-gradient capillary is the "
            "reference's after two fixed steps, and so is every statistic: energy, "
            "peak power, electron density, peak ionisation rate and peak intensity, "
            "with the quadrature error nothing on both sides.",
            bound_note="This run is strongly ionising: each quantity is bounded by "
            "its own measured response to a one-ulp launch perturbation times the "
            "count of rounding seeds, at least the retired 2.16e-12. Measured: the "
            "field 7.9e-16 against 6.6e-12, the energy 6.9e-16 against 2.16e-12, "
            "the peak intensity 3.6e-16 against 3.4e-12, the peak power, electron "
            "density and peak rate exactly the reference's. The field is new, the "
            "old test compared only the statistics; and the old test let "
            "pytest.approx's default absolute 1e-12 hold the energy (2e-4 J, so 5e-9 "
            "of it), which the row does not.",
            build=partial(
                ionising_capillary,
                statistics=(*IONISING_STATISTICS, "quadrature_error"),
            ),
            switch_off=without_plasma,
            rungs={"steps": field_response},
            every_statistic=held_to_their_response,
            metric="peak",
        ),
        column_row(
            name="ionising capillary trapezoid statistics at the launch",
            route="capillary (pressure gradient)",
            response="kerr, plasma",
            features="field grid, trapezoid rule, launch statistics",
            why="Under the trapezoid rule native reads the ionising capillary's "
            "electron density and peak ionisation rate at the launch as the "
            "reference's own scan of the same state.",
            bound_note="The two synthesise the one state, so they are held normwise "
            "to matched.stash_bound, derived for a stashed scan against a fresh one "
            "(1.6e-11 here). Measured: both exactly the reference's.",
            build=partial(
                ionising_capillary,
                RunningIntegral.trapezoid(),
                ("electron_density", "peak_ionisation_rate"),
            ),
            switch_off=without_plasma,
            rungs={"launch": launch_stash},
        ),
        column_row(
            name="one-tap band mode_averaged",
            route="capillary (mode-averaged)",
            response="plasma",
            features="band rule of one tap",
            why="On the mode-averaged helium capillary the band rule of one tap, "
            "K = 0, is the reference's in one evaluation at the launch.",
            build=partial(one_tap, "mode_averaged"),
            switch_off=without_plasma,
            rungs={"derivative": one_tap_budget},
            metric="peak",
            vacuous=HELIUM_FREES_NOTHING_AT_THE_LAUNCH,
        ),
        column_row(
            name="one-tap band radial",
            route="radial",
            response="plasma",
            features="band rule of one tap",
            why="On argon on the radial route the band rule of one tap, K = 0, is "
            "the reference's in one evaluation at the launch.",
            bound_note="The retired 64 n_to eps for the one tap (1.1e-11 here). "
            "Measured 3.1e-16; native's own test of this rule compares two native "
            "columns and is marked for a device only.",
            build=partial(one_tap, "radial"),
            switch_off=without_plasma,
            rungs={"derivative": one_tap_budget},
            metric="peak",
        ),
        column_row(
            name="envelope harmonic",
            route="gnlse",
            response="kerr",
            features="envelope grid, third harmonic, seeded launch",
            why="On an envelope grid native's third-harmonic Kerr column is the "
            "reference's in one evaluation at a launch that fills the band.",
            bound_note="The free ladder's 32 eps per transform level (8.5e-14 here), "
            "the retired column's 8 eps with the transforms native's first stage "
            "takes. Measured 4.4e-16.",
            build=harmonic_column,
            switch_off=without_harmonic,
            rungs={"derivative": transform_levels},
            metric="peak",
        ),
    )

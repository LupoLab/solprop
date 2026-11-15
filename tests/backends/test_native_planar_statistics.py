# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Planar statistics and the hard mask on native, against numpy_ref (SEP 43, 33, 35).

**Every statistic on one state.** The Numba test read every spatial statistic and each
plasma one, ``species_*`` included, at the reference's final state of three hybrid
cases, the width that three y samples cannot resolve being NaN on both. Each plasma
statistic is held to its own measured response to a one-ulp launch perturbation times
one evaluation's rounding in ulps, plus a scan, which is the original's bound. The
survivor read only ``electron_density`` and ``peak_ionisation_rate`` on a different
family at another bound.

**The hard mask on a slab.** The split-step finish's last pass stores the rows through
the mask, so inactive frequencies hold exact zeros, a structural contract (class P). The
Numba test called the kernel alone on small integers; this hands native's whole finish
rows with content off the band, on the real collar case, which is the same input to a
wider call.
"""

import math

import numpy as np
import pytest

from backends.planar_cases import planar_case
from backends.planar_plate_cases import chunked
from backends.planar_plate_cases import planar_case as collar_family_case
from bounds import assert_within, fft_bound, gamma
from sessions.runs import launch_statistics, load_native
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.stats import StatsSpec

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

EPS = float(np.finfo(float).eps)
UNIT = EPS / 2
PLASMA_CASES = ("field-plasma", "envelope-plasma")
FIELD_STATISTICS = (
    "energy",
    "peak_intensity",
    "on_axis_intensity",
    "on_axis_spectrum",
    "beam_radius",
    "fwhm_r",
)
PLASMA_STATISTICS = (
    "electron_density",
    "peak_ionisation_rate",
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
)


def finish_bound(problem):
    """Return the finish's bound: two transform pairs over the whole of the rows."""
    return 2 * fft_bound(int(np.prod(problem.shape)))


def evaluation_bound(name, grid, space):
    """Return one evaluation's relative bound: the transforms, the x GEMMs, a scan."""
    depth = math.ceil(math.log2(grid.n_to)) + math.ceil(math.log2(space.n_y))
    bound = 32 * EPS * depth + 2 * space.n_p * gamma(space.x.size)
    if name in PLASMA_CASES:
        bound += gamma(12 * grid.n_to)
    return bound


def run(backend_class, problem, run_request, grid, length):
    """Propagate the problem's launch and return the state."""
    backend = backend_class(problem, run_request, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(length)
        return backend.observe().require_field().copy()
    finally:
        backend.close()


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("kerr", marks=pytest.mark.medium),
        # slow by cost (#792): ~50 s on CI macOS and 74-106 s on Windows x64; a parity
        # check, not a witness
        pytest.param("field-plasma", marks=pytest.mark.slow),
        pytest.param("envelope-plasma", marks=pytest.mark.medium),
    ],
)
def test_each_statistic_is_the_reference_s_on_the_same_state(name):
    """Every spatial statistic, and each plasma one, at the reference's final state.

    The field statistics are one shared host function on both backends
    (:func:`solprop.backends.spatial.collect_spatial`, synthesising the node volume
    itself), so they agree to that function's own reductions, the Cartesian statistics'
    ``gamma``. The plasma statistics are native's columns against the reference's passes
    on nodes that differ by one evaluation's rounding; each is bounded by its own
    measured response to a one-ulp launch perturbation (the largest of three random-sign
    draws) times that rounding in ulps, plus ``gamma(n_to)`` for either side's own scan.
    """
    grid, space, problem, run_request = planar_case(name)
    plasma = name in PLASMA_CASES
    names = FIELD_STATISTICS + (PLASMA_STATISTICS if plasma else ())
    spec = StatsSpec(names=names)
    state = run(NumpyRefBackend, problem, run_request, grid, grid.length)

    request = run_request.for_run(stats=spec)

    def statistics(backend_class, field):
        backend = backend_class(problem, request, grid=grid)
        try:
            return launch_statistics(backend, problem, request, field=field)
        finally:
            backend.close()

    theirs = statistics(NumpyRefBackend, state)
    ours = statistics(NativeBackend, state)
    assert set(ours) == set(theirs) == set(names)
    reductions = gamma(128 * (space.n_columns + space.n_nodes) + 32 * grid.n_to)
    for field_name in FIELD_STATISTICS:
        # A NaN must be on both: three y samples cannot resolve a width.
        np.testing.assert_allclose(
            ours[field_name], theirs[field_name], rtol=reductions, err_msg=field_name
        )
    assert float(np.ravel(ours["energy"])[0]) > 0
    if space.n_y == 3:
        assert np.any(np.isnan(theirs["fwhm_r"]))
    if not plasma:
        return
    generator = np.random.default_rng(49)
    response = dict.fromkeys(PLASMA_STATISTICS, 0.0)
    for _ in range(3):
        signs = generator.choice([-1.0, 1.0], size=state.shape)
        moved = statistics(NumpyRefBackend, state * (1 + UNIT * signs))
        for stat in PLASMA_STATISTICS:
            scale = np.maximum(np.abs(theirs[stat]), np.max(np.abs(theirs[stat])))
            change = np.max(np.abs(moved[stat] - theirs[stat]) / scale)
            response[stat] = max(response[stat], float(change))
    seeds = evaluation_bound(name, grid, space) / UNIT
    scan = gamma(2 * grid.n_to)
    for stat in PLASMA_STATISTICS:
        scale = np.max(np.abs(theirs[stat]))
        assert scale > 0, stat
        bound = seeds * response[stat] + scan
        # The ionised fraction is a share of one: its rounding is one unit roundoff
        # absolute however small the share (Linux CI measured 2**-53 at a share of
        # 1.7e-5 after SEP 40's ramp moved the final state).
        floor = 2 * UNIT if stat == "species_ionisation" else 0.0
        assert np.max(np.abs(ours[stat] - theirs[stat])) <= bound * scale + floor, stat


def finished(problem, run_request, grid, rows, dz, options=None):
    """Return each backend's finish of the rows it is handed, the reference's first."""
    reference = NumpyRefBackend(problem, run_request, grid=grid, options=options)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)
        expected = np.array(reference._finish(rows[None].copy(), dz))[0]
    finally:
        reference.close()
    backend = NativeBackend(problem, run_request, grid=grid, options=options)
    try:
        native = load_native(backend, problem.initial.field0, clear_fsal=False)
        native.adopt(np.ascontiguousarray(rows.reshape(-1)), dz)
        actual = native.field_to_host().reshape(problem.shape)
    finally:
        backend.close()
    return expected, actual


def _split_step_finish(off_band: complex):
    """Finish the split-step collar case's launch rows set to `off_band` off the band.

    Returns the problem, its active mask and the reference's and native's finished
    rows (:func:`finished`) over one step of the schedule.
    """
    grid, space, problem, run_request = collar_family_case("collar_split_step")
    active = np.asarray(problem.grids.active_mask)
    rows = np.array(problem.initial.field0, dtype=complex)
    rows[..., ~active] = off_band
    expected, actual = finished(
        problem,
        run_request,
        grid,
        rows,
        float(run_request.stepper.schedule.max_step),  # pyright: ignore[reportOptionalMemberAccess]
        chunked(space),
    )
    return problem, active, expected, actual


def test_the_hard_mask_takes_the_rows_to_the_state():
    """The split-step finish leaves exact zeros off the band, on native and the oracle.

    A load masks its field, so the rows with content off the band reach the finish
    directly: ``adopt`` on native, ``_finish`` on the reference, each the whole step's
    finish on rows it is handed. Off the band the zeros are exact (class P,
    ``docs/dev/testing.md`` section 3): any other value would be real content.
    """
    _, active, expected, actual = _split_step_finish(1.0 + 1.0j)
    assert not np.all(active)
    np.testing.assert_array_equal(expected[..., ~active], 0.0)
    np.testing.assert_array_equal(actual[..., ~active], 0.0)
    assert np.max(np.abs(actual[..., active])) > 0.0


def test_the_finish_on_the_band_is_the_reference_s():
    """The same finish on rows with nothing off the band, as every march's are."""
    problem, _, expected, actual = _split_step_finish(0.0)
    assert_within(actual, expected, finish_bound(problem), what="the finish")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "SEP 43 inventory 35: native's split-step finish masks only after the temporal "
        "absorber; the reference also masks before it, so content off the band leaks "
        "onto it (normwise departure 2.2e-8 against a bound of 3.2e-12). No march "
        "hands it such rows"
    ),
)
def test_a_finish_with_content_off_the_band_is_the_reference_s_on_it():
    """The reference masks before the collar; off-band rows show whether native does.

    Reached only by handing the finish such rows, ``adopt`` directly: a load masks, the
    source's prefactor and the in-stage term are zero there, and the flow is the
    identity there. Measured on Linux x86-64 at unit content off the band: a departure
    of 2.186e-8 relative to the norm of the reference finish, whose peak is 4.4e6 V/m.
    """
    problem, _, expected, actual = _split_step_finish(1.0 + 1.0j)
    assert_within(actual, expected, finish_bound(problem), what="the finish")

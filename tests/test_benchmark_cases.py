# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the benchmark cases' definitions without timing them (SEP 38)."""

import math
import warnings

import numpy as np
import pytest
from benchmarks.builders import build, capillary, gnlse, spatial_grids
from benchmarks.records import SuiteError
from benchmarks.registry import CASES, SEED, STEPPERS, select

from solprop.driver.simulation import CAPILLARY_RTOL, DEFAULT_RTOL
from solprop.exceptions import MaterialRangeWarning, SolpropWarning
from solprop.nonlinear.responses import PlasmaResponse
from solprop.pulses import SechPulse
from solprop.stepper import LowStorageRungeKutta, Multistep, RungeKutta


def configuration(name, stepper="default", fraction=1.0):
    return CASES[name].configuration(stepper, 1, "numpy_ref", fraction)


@pytest.mark.parametrize(
    ("name", "integrator", "rtol"),
    [
        ("gnlse_raman_shock", RungeKutta, DEFAULT_RTOL),
        ("rdw", Multistep, CAPILLARY_RTOL),
        ("he_rdw_averaged", Multistep, CAPILLARY_RTOL),
    ],
)
def test_cases_run_with_the_interface_defaults_and_a_fixed_seed(name, integrator, rtol):
    simulation = build(configuration(name))
    assert isinstance(simulation.settings.integrator, integrator)
    assert simulation.settings.controller is not None
    assert simulation.settings.controller.rtol == rtol
    assert simulation.noise is not None
    assert simulation.noise.seed == SEED
    assert simulation.noise.model == "input"


@pytest.mark.parametrize(
    ("stepper", "integrator"),
    [
        ("multistep", Multistep),
        ("multistep-interaction", Multistep),
        ("rk", RungeKutta),
    ],
)
def test_a_stepper_override_replaces_the_integrator_only(stepper, integrator):
    default = build(configuration("he_rdw_averaged"))
    chosen = build(configuration("he_rdw_averaged", stepper))
    assert isinstance(chosen.settings.integrator, integrator)
    assert chosen.settings.controller is not None
    assert default.settings.controller is not None
    assert chosen.settings.controller.rtol == default.settings.controller.rtol
    assert chosen.settings.controller.atol == default.settings.controller.atol
    if stepper == "multistep-interaction":
        assert isinstance(chosen.settings.integrator, Multistep)
        assert chosen.settings.integrator.formulation == "interaction"


def test_runs_save_evenly_to_their_stop_and_a_fraction_stops_early():
    case = CASES["argon_blueshift"]
    full = build(configuration("argon_blueshift"))
    np.testing.assert_allclose(
        full.settings.save_planes, np.linspace(0, case.length, case.planes + 1)[1:]
    )
    short = build(configuration("argon_blueshift", fraction=0.25))
    assert short.settings.save_planes[-1] == pytest.approx(case.length / 4)
    assert short.grid.length == case.length


def test_fixed_step_cases_run_their_schedule_and_refuse_overrides():
    case = CASES["kerr_beam_192"]
    assert case.steppers == ("default",)
    with pytest.raises(SuiteError, match="admits only default"):
        case.configuration("rk", 1)
    # The case keeps SEP 28's ultraviolet extrapolation of the silica fit.
    with pytest.warns(MaterialRangeWarning, match="Malitson"):
        simulation = build(configuration("kerr_beam_192"))
    assert isinstance(simulation.settings.integrator, LowStorageRungeKutta)
    schedule = simulation.settings.schedule
    assert schedule is not None
    assert len(schedule.steps) == 4
    np.testing.assert_allclose(schedule.edges[-1], case.length, rtol=1e-12)
    assert simulation.settings.fields == "planes"


@pytest.mark.parametrize(
    ("resolved", "averaged"),
    [
        ("rdw_one_mode", "he_rdw_averaged"),
        ("argon_blueshift", "argon_blueshift_averaged"),
        ("h2_raman_6ps", "h2_raman_6ps_averaged"),
    ],
)
def test_averaging_is_a_distinct_route_with_the_original_grid(resolved, averaged):
    first, second = capillary(resolved), capillary(averaged)
    assert first.settings.quadrature is not None
    assert second.settings.quadrature is None
    assert second.settings.components is None
    # A route change must not change the launch, guide or temporal axes.
    assert first.grid.length == second.grid.length
    np.testing.assert_array_equal(first.grid.omega, second.grid.omega)
    assert first.settings.stats != second.settings.stats
    if resolved == "h2_raman_6ps":
        assert not any(
            isinstance(response, PlasmaResponse) for response in first.responses
        )


def test_the_reduced_ctc_case_is_the_fundamental_alone():
    # The reduced comb-to-continuum case of the tests; its two-mode twin keeps the
    # same window and guide.
    one, two = capillary("ctc_reduced"), capillary("ctc_reduced_two_mode")
    assert (len(one.modes), len(two.modes)) == (1, 2)
    assert one.grid.length == two.grid.length == 0.1
    np.testing.assert_array_equal(one.grid.omega, two.grid.omega)


def test_the_raman_shock_case_launches_the_published_pulse():
    # Dudley, Genty and Coen's supercontinuum: a 50 fs FWHM sech, T0 = 28.4 fs.
    launch = gnlse("gnlse_raman_shock").initial
    assert isinstance(launch, SechPulse)
    assert launch.fwhm == pytest.approx(50e-15, rel=1e-15)
    assert launch.peak_power == pytest.approx(10e3, rel=1e-15)


# Building a Cartesian case allocates its whole launch (12 and 35 GB at the peak for
# the 384- and 768-square Kerr beams), so free-space sizes are read from the builders'
# grids; the two solved-mode guides solve their modes (5 and 8 GB) and are left out.
SOLVED_GUIDES = ("grin_nonlinear", "moving_solid_guide")


@pytest.mark.parametrize("name", [name for name in CASES if name not in SOLVED_GUIDES])
def test_declared_dimensions_follow_the_builders(name):
    case = CASES[name]
    if case.builder in ("kerr", "spatial") and name != "planar":
        grid, space = spatial_grids(name)
        columns, modes = space.n_columns, None
    else:
        # Some builders keep their sources' model warnings (an ultraviolet
        # extrapolation, a guide shorter than the absorber's walk-off); only the sizes
        # are read here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SolpropWarning)
            simulation = build(configuration(name))
        grid, modes = simulation.grid, len(simulation.modes)
        transverse = simulation.spatial_grid
        columns = None if transverse is None else transverse.n_columns
    assert case.dimensions[-2:] == (grid.n_t, grid.n_to)
    leading = case.dimensions[:-2]
    if columns is not None:
        assert math.prod(leading) == columns
    else:
        assert leading[0] == modes


def test_the_envelope_case_differs_from_its_field_twin_only_in_the_grid():
    field = capillary("argon_blueshift")
    envelope = capillary("argon_blueshift_envelope")
    assert field.grid.length == envelope.grid.length
    assert type(field.grid).__name__ == "FieldGrid"
    assert type(envelope.grid).__name__ == "EnvelopeGrid"


def test_the_tracking_set_is_the_default_selection():
    tracked = select()
    assert [case.name for case in tracked] == [
        "gnlse_raman_shock",
        "gnlse_soliton",
        "rdw",
        "he_rdw_averaged",
        "argon_blueshift",
        "argon_blueshift_averaged",
        "argon_blueshift_envelope",
        "h2_raman_averaged",
        "ctc",
        "radial_filament_256",
        "grin_nonlinear",
        "kerr_beam_384",
        "kerr_beam_192",
    ]
    # Every tracked width has a cost, so the default plan has an estimate.
    assert all(
        case.tracked and case.cost[0 if width == "1" else 1] is not None
        for case in tracked
        for width in case.widths
    )
    assert {case.name for case in select(all_cases=True)} == set(CASES) - {"smoke"}
    assert len({case.digest for case in CASES.values()}) == len(CASES)
    assert STEPPERS[0] == "default"


def test_selection_and_builders_refuse_unknown_names():
    with pytest.raises(SuiteError, match="unknown case"):
        select(("absent",))
    with pytest.raises(SuiteError, match="unknown set"):
        select(sets=("absent",))
    with pytest.raises(SuiteError, match="fraction"):
        CASES["rdw"].configuration("default", 1, fraction=0.0)
    with pytest.raises(SuiteError, match="not implemented"):
        capillary("absent")
    with pytest.raises(SuiteError, match="not implemented"):
        gnlse("absent")


def test_tracked_definitions_change_only_with_a_new_revision():
    # A changed definition must raise its revision: rows compare within one. The
    # digest covers the registry's declared fields; builder edits need the same.
    # Every case starts at revision 1 with the first history rows, at the release.
    assert [(case.name, case.revision, case.digest[:12]) for case in select()] == [
        ("gnlse_raman_shock", 1, "3da05c17d16e"),
        ("gnlse_soliton", 1, "46cd5ec245e2"),
        ("rdw", 1, "dbbc8e09cd88"),
        ("he_rdw_averaged", 1, "f9ebec0f93a7"),
        ("argon_blueshift", 1, "a48ead7cf1dc"),
        ("argon_blueshift_averaged", 1, "06b8b09e4444"),
        ("argon_blueshift_envelope", 1, "0b376ed0c4ae"),
        ("h2_raman_averaged", 1, "4b7440bfae07"),
        ("ctc", 1, "5e88b5b3b60f"),
        ("radial_filament_256", 2, "a7bd268f25df"),
        ("grin_nonlinear", 1, "5a1ddf84368a"),
        ("kerr_beam_384", 1, "88c44c3d288c"),
        ("kerr_beam_192", 1, "1c8baa576a04"),
    ]
    # Revision 2 for SEP 40: spatial_absorber requests a collar, and the radial
    # filaments' wall moves out to 1.5 mm, which their light no longer reaches.
    revisions = {case.name: case.revision for case in CASES.values()}
    for name in ("spatial_absorber", "radial_filament_256", "radial_filament_512"):
        assert revisions.pop(name) == 2, name
    assert set(revisions.values()) == {1}

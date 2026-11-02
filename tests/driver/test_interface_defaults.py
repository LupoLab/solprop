# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Each interface's defaults (SEP 37): integrator, tolerances, noise, edge watch.

Every route in both precisions, through its ``Simulation.from_*`` constructor and its
``prop_*`` function, and the refusals that name the option at fault.
"""

import dataclasses
import inspect
import json
from typing import TypedDict

import pytest

from driver.test_tdse_entry import _capillary, _sae
from modal_provider_cases import GENERAL_Z_END, general_grid, general_guide
from solprop.driver import Simulation, SimulationSettings, interface
from solprop.driver.defaults import (
    ROUTE_DEFAULTS,
    SINGLE_PRECISION_NOISE,
    default_integrator,
    default_rtol,
    default_scheme,
    resolve_noise,
    route_defaults,
    rtol_in_precision,
)
from solprop.driver.simulation import (
    CAPILLARY_RTOL,
    DEFAULT_RTOL,
    FREE_ATOL,
    FREE_NOISE,
    FREE_RTOL,
    SLAB_NOISE,
    SPATIAL_ATOL,
    SPATIAL_RTOL,
    TDSE_NOISE,
    WAVEGUIDE_RTOL,
    _admit_tdse_execution,
)
from solprop.exceptions import DriverError, SolpropWarning
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import MarcatiliMode
from solprop.modes.slab import PlanarGrid, SlabMode
from solprop.monitor import DEFAULT_BOUNDARY_CHECK, BoundaryCheck
from solprop.nonlinear.responses import KerrResponse
from solprop.nonlinear.tdse import TDSEExecution
from solprop.output.memory import MemoryOutput
from solprop.output.schema import DefaultsResolution, NoiseResolution, RunMetadata
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import Multistep, PIController, RungeKutta
from solprop.translate.precision import (
    DOUBLE,
    SINGLE_RTOL_CAP,
    SINGLE_RTOL_FLOOR,
    PrecisionName,
)

GNLSE = {
    "wavelength": 835e-9,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 1e-12,
    "width": 100e-15,
    "peak_power": 9.8,
    "shape": "sech",
    "n_save": 3,
    "noise": None,
}
PULSE = {"wavelength": 800e-9, "fwhm": 30e-15, "energy": 1e-6, "noise": None}
WINDOW = {"wavelength_limits": (400e-9, 2e-6), "trange": 200e-15, "n_save": 2}


def test_the_gnlse_takes_the_runge_kutta_march_and_its_tolerance():
    simulation = Simulation.from_gnlse(0.11, 0.01, [0.0, 0.0, -1.183e-26], **GNLSE)
    assert isinstance(simulation.settings.integrator, RungeKutta)
    controller = simulation.settings.controller
    assert controller is not None
    assert controller.rtol == DEFAULT_RTOL == 1e-7
    named = Simulation.from_gnlse(
        0.11, 0.01, [0.0, 0.0, -1.183e-26], integrator=Multistep(), **GNLSE
    )
    assert isinstance(named.settings.integrator, Multistep)


def _stepping(simulation):
    controller = simulation.settings.controller
    assert controller is not None
    return type(simulation.settings.integrator), controller.rtol


def test_the_capillary_takes_the_adams_march_whatever_its_representation():
    averaged = Simulation.from_capillary(
        100e-6, 0.01, "He", 1.0, mode_average=True, **PULSE, **WINDOW
    )
    assert _stepping(averaged) == (Multistep, CAPILLARY_RTOL)
    single = Simulation.from_capillary(100e-6, 0.01, "He", 1.0, **PULSE, **WINDOW)
    assert _stepping(single) == (Multistep, CAPILLARY_RTOL)
    multimode = Simulation.from_capillary(
        100e-6, 0.01, "He", 1.0, modes=2, **PULSE, **WINDOW
    )
    assert _stepping(multimode) == (Multistep, CAPILLARY_RTOL)
    assert _stepping(_capillary(tdse=_sae())) == (Multistep, CAPILLARY_RTOL)
    named = Simulation.from_capillary(
        100e-6, 0.01, "He", 1.0, integrator=RungeKutta(), rtol=1e-5, **PULSE, **WINDOW
    )
    assert _stepping(named) == (RungeKutta, 1e-5)


def test_the_waveguide_takes_the_runge_kutta_march_and_its_tolerance():
    simulation = Simulation.from_waveguide(
        [MarcatiliMode(100e-6)], 0.01, "He", 1.0, **PULSE, **WINDOW
    )
    assert isinstance(simulation.settings.integrator, RungeKutta)
    controller = simulation.settings.controller
    assert controller is not None
    assert controller.rtol == WAVEGUIDE_RTOL == 1e-5


def _spatial(space, controller=None, integrator=None, noise=None, copied=False):
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    settings = SimulationSettings(
        controller,
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
        integrator=integrator,
    )
    if copied:
        settings = dataclasses.replace(settings, ionisation_field_range=(1e7, 1e11))
    arguments = (
        SpatialMedium(),
        [],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6),
    )
    if isinstance(space, RadialGrid):
        return Simulation.from_radial(
            grid, space, *arguments, settings=settings, noise=noise
        )
    return Simulation.from_free(grid, space, *arguments, settings=settings, noise=noise)


def test_a_radial_run_without_a_controller_takes_the_spatial_default():
    controller = _spatial(RadialGrid(6e-4, 16)).settings.controller
    assert controller is not None
    assert (controller.rtol, controller.atol) == (SPATIAL_RTOL, SPATIAL_ATOL)
    explicit = PIController(rtol=1e-7, atol=1e-12)
    assert _spatial(RadialGrid(6e-4, 16), explicit).settings.controller is explicit


def test_a_free_space_run_takes_the_runge_kutta_march_and_no_noise():
    simulation = _spatial(FreeGrid(3e-4, 8, 3e-4, 8))
    assert isinstance(simulation.settings.integrator, RungeKutta)
    controller = simulation.settings.controller
    assert controller is not None
    assert (controller.rtol, controller.atol) == (FREE_RTOL, FREE_ATOL)
    named = _spatial(FreeGrid(3e-4, 8, 3e-4, 8), integrator=Multistep())
    assert isinstance(named.settings.integrator, Multistep)
    quiet = _spatial(FreeGrid(3e-4, 8, 3e-4, 8), noise="default")
    assert quiet.noise is None
    assert quiet.noise_resolution == NoiseResolution("default", None, FREE_NOISE)
    radial = _spatial(RadialGrid(6e-4, 16), noise="default")
    assert radial.noise_resolution == NoiseResolution("default", "input")


def test_a_copy_of_the_settings_still_names_no_integrator():
    # dataclasses.replace passes the resolved default back in; the copy must still
    # read as unnamed, or a free-space run would keep the Multistep default.
    space = FreeGrid(3e-4, 8, 3e-4, 8)
    copied = _spatial(space, copied=True)
    assert isinstance(copied.settings.integrator, RungeKutta)
    named = _spatial(space, integrator=Multistep(), copied=True)
    assert isinstance(named.settings.integrator, Multistep)
    settings = SimulationSettings(save_planes=[1.0], dz_max=1.0, dz_init=1e-3)
    assert not dataclasses.replace(settings, step_planes=2).integrator_named
    assert dataclasses.replace(settings, integrator=Multistep()).integrator_named
    single = dataclasses.replace(settings, precision="single", backend="numpy_ref")
    assert isinstance(single.integrator, RungeKutta)
    assert not single.integrator_named


def test_a_simulation_built_directly_needs_a_controller():
    built = Simulation.from_gnlse(0.11, 0.01, [0.0, 0.0, -1.183e-26], **GNLSE)
    settings = SimulationSettings(
        save_planes=built.settings.save_planes,
        dz_max=built.settings.dz_max,
        dz_init=built.settings.dz_init,
    )
    assert "interface default tolerances" in repr(settings)
    with pytest.raises(DriverError, match="needs a PIController"):
        Simulation(
            built.grid, built.modes, built.responses, built.initial, settings=settings
        )


def test_a_tdse_run_takes_no_noise_by_default():
    simulation = _capillary(tdse=_sae(), noise="default")
    assert simulation.noise is None
    assert simulation.noise_resolution == NoiseResolution("default", None, TDSE_NOISE)
    with pytest.raises(DriverError, match="takes no noise by default"):
        _capillary(tdse=_sae(), noise="default", seed=3)


def test_noise_beside_the_tdse_is_warned_on():
    with pytest.warns(SolpropWarning, match="TDSE"):
        simulation = _capillary(tdse=_sae(), noise="input", seed=3)
    assert simulation.noise is not None


# ------------------------------------------- every route in both precisions


class Arithmetic(TypedDict, total=False):
    """The keywords that put a run in single precision, typed for unpacking."""

    precision: PrecisionName
    backend: str


SINGLE: Arithmetic = {"precision": "single", "backend": "numpy_ref"}
"""Single precision on the oracle, which every machine has (MLX is a Mac's alone)."""

PLANAR = EnvelopeGrid.from_limits(1e-3, 800e-9, (650e-9, 1000e-9), 600e-15)
SPATIAL = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)


def _keyword(precision) -> Arithmetic:
    """Return the keyword arguments of a keyword interface in `precision`."""
    return {} if precision == "double" else SINGLE


def _settings(grid, precision):
    """Return spatial settings without a controller, so the interface's applies."""
    return SimulationSettings(
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
        **_keyword(precision),
    )


def _beam():
    return SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6
    )


def _planar_launch():
    pulse = GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3)
    return PlanarPulse(pulse, 400e-6, waist_x=0.735 * 63.5e-6)


def _slabs():
    return [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)]


def _general(precision, entry):
    grid = general_grid()
    return entry(
        (),
        GENERAL_Z_END,
        provider=general_guide(grid=grid),
        grid=grid,
        wavelength=835e-9,
        fwhm=20e-15,
        energy=1e-12,
        shape="gauss",
        n_save=2,
        noise=None,
        **_keyword(precision),
    )


GNLSE_ARGS = (0.11, 0.01, [0.0, 0.0, -1.183e-26])
CAPILLARY_ARGS = (100e-6, 0.01, "He", 1.0)
WAVEGUIDE_ARGS = ([MarcatiliMode(100e-6)], 0.01, "He", 1.0)


def _build(route, precision, through):  # noqa: PLR0911 - one return per route family
    """Build `route` in `precision` through its ``"from"`` constructor or ``"prop"``.

    Under the `intercepted` fixture a ``prop_*`` function returns a placeholder
    output; :func:`_built` then finds the Simulation it built.
    """
    keywords = _keyword(precision)
    fibre = {**PULSE, **WINDOW, **keywords}
    constructors = {
        "from": {
            "gnlse": Simulation.from_gnlse,
            "capillary": Simulation.from_capillary,
            "waveguide": Simulation.from_waveguide,
            "radial": Simulation.from_radial,
            "free": Simulation.from_free,
            "planar": Simulation.from_planar,
        },
        "prop": {
            "gnlse": interface.prop_gnlse,
            "capillary": interface.prop_capillary,
            "waveguide": interface.prop_waveguide,
            "radial": interface.prop_radial,
            "free": interface.prop_free,
            "planar": interface.prop_planar,
        },
    }[through]
    family, _, variant = route.partition(" ")
    entry = constructors[family]
    if family == "gnlse":
        return entry(*GNLSE_ARGS, **GNLSE, **keywords)
    if family == "capillary":
        extra = {
            "": {},
            "modes=2": {"modes": 2},
            "mode_average": {"mode_average": True},
            "envelope": {"envelope": True},
        }[variant]
        return entry(*CAPILLARY_ARGS, **fibre, **extra)
    if family == "waveguide":
        if variant == "general":
            return _general(precision, entry)
        extra = {"mode_average": True} if variant == "mode_average" else {}
        return entry(*WAVEGUIDE_ARGS, **fibre, **extra)
    if family == "radial":
        return entry(
            SPATIAL,
            RadialGrid(6e-4, 16),
            SpatialMedium(),
            [],
            _beam(),
            settings=_settings(SPATIAL, precision),
        )
    if family == "free":
        space = (
            LineGrid(3e-4, 16, window_factor=0)
            if variant == "line"
            else FreeGrid(3e-4, 12, 4e-4, 16, window_factor=0)
        )
        return entry(
            SPATIAL,
            space,
            SpatialMedium(),
            [],
            _beam(),
            settings=_settings(SPATIAL, precision),
        )
    return entry(
        PLANAR,
        PlanarGrid(_slabs(), 1e-3, 8),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        _planar_launch(),
        settings=_settings(PLANAR, precision),
    )


# route: (binary64 integrator, binary64 rtol, binary32 rtol). Binary32 steps with
# RungeKutta() everywhere; its rtol is the larger of binary64's and the floor 1e-6,
# capped at 1e-5 (the maintainers' ruling on the single-precision default).
DEFAULTS = {
    "gnlse": (RungeKutta, 1e-7, 1e-6),
    "capillary": (Multistep, 1e-6, 1e-6),
    "capillary modes=2": (Multistep, 1e-6, 1e-6),
    "capillary mode_average": (Multistep, 1e-6, 1e-6),
    "capillary envelope": (Multistep, 1e-6, 1e-6),
    "waveguide": (RungeKutta, 1e-5, 1e-5),
    "waveguide mode_average": (RungeKutta, 1e-5, 1e-5),
    "waveguide general": (RungeKutta, 1e-5, 1e-5),
    "radial": (Multistep, 1e-5, 1e-5),
    "free cartesian": (RungeKutta, 1e-4, 1e-5),
    "free line": (RungeKutta, 1e-4, 1e-5),
    "planar": (Multistep, 1e-5, 1e-5),
}


@pytest.fixture
def intercepted(monkeypatch):
    """Keep every Simulation a prop_* function would run, and run none of them."""
    built = []

    def run(self, *args, **kwargs):
        built.append(self)
        return MemoryOutput()

    monkeypatch.setattr(Simulation, "run", run)
    return built


def _built(result, intercepted):
    """Return the Simulation a constructor returned or a prop_* function built."""
    return result if isinstance(result, Simulation) else intercepted[-1]


@pytest.mark.parametrize("through", ["from", "prop"])
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("route", list(DEFAULTS))
def test_every_route_resolves_its_default_integrator_and_tolerance(
    route, precision, through, intercepted
):
    simulation = _built(_build(route, precision, through), intercepted)
    double_integrator, double_rtol, single_rtol = DEFAULTS[route]
    integrator, rtol = _stepping(simulation)
    # The resolved defaults are exact structural values, not computed ones.
    if precision == "double":
        assert (integrator, rtol) == (double_integrator, double_rtol)
    else:
        assert (integrator, rtol) == (RungeKutta, single_rtol)
    assert not simulation.settings.integrator_named


@pytest.mark.parametrize("route", list(DEFAULTS))
def test_a_copy_in_the_other_precision_takes_the_route_s_default(route):
    """An interface's default is not a named integrator, whatever the copy changes."""
    single = _build(route, "single", "from").settings
    double = dataclasses.replace(single, precision="double")
    assert type(double.integrator) is DEFAULTS[route][0]
    assert not double.integrator_named
    back = dataclasses.replace(double, precision="single")
    assert isinstance(back.integrator, RungeKutta)


def test_the_single_precision_tolerance_rule():
    assert [rtol_in_precision(rtol, "single") for rtol in (1e-8, 1e-7, 1e-6)] == [
        SINGLE_RTOL_FLOOR
    ] * 3
    assert [rtol_in_precision(rtol, "single") for rtol in (1e-5, 1e-4, 1e-3)] == [
        SINGLE_RTOL_CAP
    ] * 3
    assert rtol_in_precision(1e-4, "double") == 1e-4
    # An explicit tolerance is the caller's, above the cap included.
    simulation = Simulation.from_waveguide(
        *WAVEGUIDE_ARGS, rtol=1e-4, **PULSE, **WINDOW, **SINGLE
    )
    assert _stepping(simulation) == (RungeKutta, 1e-4)


# ------------------------------------------------- refusals name their option


def test_single_precision_refuses_the_adams_method_by_its_own_name():
    for backend in ("numpy_ref", "mlx"):
        with pytest.raises(DriverError, match=r"Adams integrator \(Multistep\) is not"):
            SimulationSettings(
                PIController(rtol=1e-6, atol=0.0),
                save_planes=[1.0],
                dz_max=1.0,
                dz_init=1e-3,
                precision="single",
                backend=backend,
                integrator=Multistep(),
            )


def test_a_tdse_run_s_refusals_name_the_precision_and_the_integrator():
    with pytest.raises(DriverError, match="precision='single' refuses a TDSE"):
        _capillary(tdse=_sae(), **SINGLE)
    with pytest.raises(DriverError, match=r"integrator=Multistep\(\).*RungeKutta"):
        _capillary(tdse=_sae(), integrator=RungeKutta())
    with pytest.raises(DriverError, match="explicit backend='numpy_ref'"):
        SimulationSettings(
            save_planes=[1.0],
            dz_max=1.0,
            dz_init=1e-3,
            backend="numba_cpu",
            tdse_execution=TDSEExecution(),
        )
    # Not a record at all: refused where the annotations are not checked at run time.
    settings = SimulationSettings(
        save_planes=[1.0], dz_max=1.0, dz_init=1e-3, backend="numpy_ref"
    )
    object.__setattr__(settings, "tdse_execution", object())
    with pytest.raises(DriverError, match="must be a TDSEExecution"):
        inspect.unwrap(_admit_tdse_execution)(settings, DOUBLE)


# ---------------------------------------------------------------- slab noise


def test_a_slab_mode_set_carries_no_noise_and_refuses_a_model():
    launch = {"wavelength": 800e-9, "fwhm": 30e-15, "energy": 1e-6}
    slabs = Simulation.from_waveguide(_slabs(), 0.01, "Ar", 1.0, **launch, **WINDOW)
    assert slabs.noise is None
    assert slabs.noise_resolution == NoiseResolution("default", None, SLAB_NOISE)
    with pytest.raises(DriverError, match="slab mode set carries no noise"):
        Simulation.from_waveguide(
            _slabs(), 0.01, "Ar", 1.0, noise="input", **launch, **WINDOW
        )
    with pytest.raises(DriverError, match="takes no noise by default"):
        Simulation.from_waveguide(_slabs(), 0.01, "Ar", 1.0, seed=3, **launch, **WINDOW)


# ------------------------------------------------- boundary_check everywhere


ENTRY_POINTS = [
    Simulation.from_gnlse,
    Simulation.from_capillary,
    Simulation.from_waveguide,
    Simulation.from_radial,
    Simulation.from_free,
    Simulation.from_planar,
    interface.prop_gnlse,
    interface.prop_capillary,
    interface.prop_waveguide,
    interface.prop_radial,
    interface.prop_free,
    interface.prop_planar,
]


@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda entry: entry.__name__)
def test_every_entry_point_takes_boundary_check(entry):
    parameter = inspect.signature(entry).parameters["boundary_check"]
    assert parameter.default is DEFAULT_BOUNDARY_CHECK


@pytest.mark.parametrize("through", ["from", "prop"])
@pytest.mark.parametrize(
    "route", ["gnlse", "capillary", "waveguide", "radial", "free cartesian", "planar"]
)
def test_boundary_check_reaches_the_settings(route, through, intercepted):
    relaxed = BoundaryCheck(budget=1.0, stop_factor=2.0)
    for check in (None, relaxed, DEFAULT_BOUNDARY_CHECK):
        simulation = _built(_with_check(route, through, check), intercepted)
        assert simulation.settings.boundary_check is check


def _with_check(route, through, check):
    """Build a double-precision `route` passing `check` as the keyword."""
    family = route.partition(" ")[0]
    if family in ("radial", "free", "planar"):
        # The spatial entry points take it beside their settings.
        entry = {
            "from": {
                "radial": Simulation.from_radial,
                "free": Simulation.from_free,
                "planar": Simulation.from_planar,
            },
            "prop": {
                "radial": interface.prop_radial,
                "free": interface.prop_free,
                "planar": interface.prop_planar,
            },
        }[through][family]
        grid = PLANAR if family == "planar" else SPATIAL
        if family == "planar":
            arguments = (
                PlanarGrid(_slabs(), 1e-3, 8),
                [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
                _planar_launch(),
            )
        else:
            space = (
                RadialGrid(6e-4, 16)
                if family == "radial"
                else FreeGrid(3e-4, 12, 4e-4, 16, window_factor=0)
            )
            arguments = (space, SpatialMedium(), [], _beam())
        return entry(
            grid,
            *arguments,
            settings=_settings(grid, "double"),
            boundary_check=check,
        )
    entry = {
        "from": {
            "gnlse": Simulation.from_gnlse,
            "capillary": Simulation.from_capillary,
            "waveguide": Simulation.from_waveguide,
        },
        "prop": {
            "gnlse": interface.prop_gnlse,
            "capillary": interface.prop_capillary,
            "waveguide": interface.prop_waveguide,
        },
    }[through][family]
    if family == "gnlse":
        return entry(*GNLSE_ARGS, **GNLSE, boundary_check=check)
    arguments = CAPILLARY_ARGS if family == "capillary" else WAVEGUIDE_ARGS
    return entry(*arguments, **PULSE, **WINDOW, boundary_check=check)


def test_a_spatial_keyword_and_settings_naming_two_checks_are_refused():
    relaxed = BoundaryCheck(budget=1.0, stop_factor=2.0)
    settings = dataclasses.replace(_settings(SPATIAL, "double"), boundary_check=None)
    with pytest.raises(DriverError, match="already declare a boundary_check"):
        Simulation.from_radial(
            SPATIAL,
            RadialGrid(6e-4, 16),
            SpatialMedium(),
            [],
            _beam(),
            settings=settings,
            boundary_check=relaxed,
        )
    # The same check named twice is no conflict, and settings' own stands alone.
    same = Simulation.from_radial(
        SPATIAL,
        RadialGrid(6e-4, 16),
        SpatialMedium(),
        [],
        _beam(),
        settings=settings,
        boundary_check=None,
    )
    assert same.settings.boundary_check is None


# --- one table of route defaults -------------------------------------------------


# The table's rows the DEFAULTS matrix above measures through the constructors, so the
# table and what each route resolves to cannot drift apart.
TABLE_ROWS = {
    "gnlse": "gnlse",
    "capillary": "capillary",
    "capillary modes=2": "capillary",
    "capillary mode_average": "capillary",
    "capillary envelope": "capillary",
    "waveguide": "waveguide",
    "waveguide mode_average": "waveguide",
    "waveguide general": "waveguide",
    "radial": "radial",
    "free cartesian": "free",
    "free line": "free",
    "planar": "planar",
}


@pytest.mark.parametrize("route", list(DEFAULTS))
def test_the_table_holds_what_every_route_resolves(route):
    row = route_defaults(TABLE_ROWS[route])
    integrator, double_rtol, single_rtol = DEFAULTS[route]
    assert row.integrator == integrator.__name__
    assert row.rtol == double_rtol
    assert default_rtol(TABLE_ROWS[route], "single") == single_rtol
    assert type(default_integrator(TABLE_ROWS[route], "double")) is integrator
    assert type(default_integrator(TABLE_ROWS[route], "single")) is RungeKutta


def test_every_row_of_the_table_is_measured_or_a_named_variant():
    assert set(ROUTE_DEFAULTS) == {
        *TABLE_ROWS.values(),
        "capillary tdse",
        "waveguide slab",
    }
    assert route_defaults("capillary tdse").noise_reason == TDSE_NOISE
    assert route_defaults("waveguide slab").noise == "unmeasured"
    with pytest.raises(DriverError, match="no route defaults for 'fibre'"):
        route_defaults("fibre")


@pytest.mark.parametrize(
    ("route", "precision", "model", "reason"),
    [
        ("gnlse", "double", "input", None),
        ("capillary", "double", "input", None),
        ("capillary tdse", "double", None, TDSE_NOISE),
        ("waveguide slab", "double", None, SLAB_NOISE),
        ("radial", "double", "input", None),
        ("free", "double", None, FREE_NOISE),
        ("planar", "double", None, route_defaults("planar").noise_reason),
        *((route, "single", None, SINGLE_PRECISION_NOISE) for route in ROUTE_DEFAULTS),
    ],
)
def test_every_route_and_precision_resolves_its_default_noise(
    route, precision, model, reason
):
    resolved, resolution = resolve_noise(route, "default", None, precision)
    assert resolved == model
    assert resolution == NoiseResolution("default", model, reason)


def test_a_model_is_refused_where_no_photon_measure_is_declared():
    for route in ("waveguide slab", "planar"):
        with pytest.raises(DriverError, match="pass noise=None"):
            resolve_noise(route, "input", None, "double")
    assert resolve_noise("free", "input", None, "double")[0] == "input"


@pytest.mark.parametrize(
    ("route", "interpolate", "scheme"),
    [
        (None, False, "nonlinear"),
        ("gnlse", False, "nonlinear"),
        ("radial", False, "split_step"),
        ("free", False, "split_step"),
        ("planar", True, "nonlinear"),
    ],
)
def test_each_route_applies_its_collar_by_its_scheme(route, interpolate, scheme):
    assert default_scheme(route, interpolate_planes=interpolate) == scheme


def _gnlse_settings(**keywords):
    return Simulation.from_gnlse(*GNLSE_ARGS, **GNLSE, **keywords).settings


def _control(settings):
    """Return the settings' controller, which the keyword routes always give."""
    assert settings.controller is not None
    return settings.controller


def test_a_default_tolerance_is_resolved_again_in_the_other_precision():
    double = _gnlse_settings(backend="numpy_ref")
    assert double.controller_defaulted
    assert _control(double).rtol == DEFAULT_RTOL
    single = dataclasses.replace(double, precision="single")
    # 1e-7 would be refused under single precision; the copy takes that default.
    assert _control(single).rtol == SINGLE_RTOL_FLOOR
    assert _control(single).atol == _control(double).atol
    assert single.controller_defaulted
    back = dataclasses.replace(single, precision="double")
    assert _control(back).rtol == DEFAULT_RTOL


def test_a_named_tolerance_is_kept_in_the_other_precision():
    named = _gnlse_settings(backend="numpy_ref", rtol=1e-5)
    assert not named.controller_defaulted
    assert _control(dataclasses.replace(named, precision="single")).rtol == 1e-5
    replaced = dataclasses.replace(
        named, controller=PIController(rtol=1e-4, atol=1e-14)
    )
    assert not replaced.controller_defaulted


def test_a_copy_in_single_precision_resolves_auto_again(monkeypatch):
    double = _gnlse_settings()
    monkeypatch.setattr("solprop.driver.simulation.metal_available", lambda: True)
    assert dataclasses.replace(double, precision="single").backend == "mlx"
    monkeypatch.setattr("solprop.driver.simulation.metal_available", lambda: False)
    # Without a Metal device, single precision's "auto" is refused by design.
    with pytest.raises(DriverError, match="backend='auto' resolves to the mlx"):
        dataclasses.replace(double, precision="single")
    # A named backend is not resolved again: binary32 refuses it by name.
    named = _gnlse_settings(backend="numpy_ref")
    assert dataclasses.replace(named, precision="single").backend == "numpy_ref"


def test_the_stepping_defaults_are_recorded_and_read_back():
    simulation = Simulation.from_gnlse(*GNLSE_ARGS, **GNLSE, backend="numpy_ref")
    record = simulation.metadata().defaults
    assert record == DefaultsResolution(
        "gnlse", ("integrator", "rtol", "absorber_scheme")
    )
    assert RunMetadata.from_json(simulation.metadata().to_json()).defaults == record
    named = Simulation.from_gnlse(
        *GNLSE_ARGS,
        **GNLSE,
        backend="numpy_ref",
        rtol=1e-6,
        integrator=Multistep(),
        absorber_scheme="nonlinear",
    )
    assert named.metadata().defaults == DefaultsResolution("gnlse", ())


def test_a_record_written_before_the_defaults_were_recorded_reads():
    simulation = Simulation.from_gnlse(*GNLSE_ARGS, **GNLSE, backend="numpy_ref")
    document = json.loads(simulation.metadata().to_json())
    del document["defaults"]
    assert RunMetadata.from_json(json.dumps(document)).defaults is None


def test_the_displays_say_which_values_are_defaults():
    settings = _gnlse_settings(backend="numpy_ref")
    text = str(settings)
    assert "RungeKutta (gnlse default)" in text
    assert "atol 1e-14 (gnlse default)" in text
    assert "nonlinear scheme (default)" in text
    free = _spatial(FreeGrid(3e-4, 8, 3e-4, 8), noise="default")
    # The block wraps long rows, so compare the words.
    assert f"none ({FREE_NOISE})" in " ".join(str(free).split())
    assert "split_step scheme (default)" in str(free.settings)


def test_an_incident_launch_records_a_default_noise_as_default():
    simulation = Simulation.from_gnlse(*GNLSE_ARGS, **{**GNLSE, "noise": "default"})
    launched = simulation.with_input(simulation._field0, noise=simulation.noise)
    assert launched.noise_resolution == simulation.noise_resolution
    assert launched.noise_resolution.requested == "default"


def test_an_interior_launch_keeps_the_single_precision_reason():
    single = Simulation.from_gnlse(
        *GNLSE_ARGS,
        **{**GNLSE, "noise": "default"},
        precision="single",
        backend="numpy_ref",
    )
    moved = single.with_input(single._field0, initial_z=0.001)
    assert moved.noise_resolution.reason == SINGLE_PRECISION_NOISE

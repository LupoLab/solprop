# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The parameter metadata agrees with the signatures it describes (SEP 41, S9)."""

import inspect
import re
import types
import typing
from pathlib import Path

import numpy as np
import pytest

from solprop.driver import Simulation
from solprop.driver.defaults import default_rtol
from solprop.driver.interface import (
    prop_capillary,
    prop_free,
    prop_gnlse,
    prop_planar,
    prop_radial,
    prop_waveguide,
)
from solprop.driver.simulation import SimulationSettings
from solprop.exceptions import DriverError
from solprop.modes import MarcatiliMode, ModeCollection
from solprop.recipe import metadata
from solprop.recipe.document import EXECUTION_KEYS, REFUSED_BUILD_KEYS
from solprop.recipe.quantity import Unit
from solprop.recipe.registry import REGISTRY

ROUTES = ("gnlse", "capillary", "waveguide", "radial", "free", "planar")
DOCS = Path(__file__).resolve().parents[2] / "docs"
BUILDERS = {
    "gnlse": Simulation.from_gnlse,
    "capillary": Simulation.from_capillary,
    "waveguide": Simulation.from_waveguide,
    "radial": Simulation.from_radial,
    "free": Simulation.from_free,
    "planar": Simulation.from_planar,
}
PROPS = {
    "gnlse": prop_gnlse,
    "capillary": prop_capillary,
    "waveguide": prop_waveguide,
    "radial": prop_radial,
    "free": prop_free,
    "planar": prop_planar,
}
# from_waveguide's memory_budget admits a guide's tables at build; prop_waveguide has
# none, so the export writes such a recipe through Simulation.from_waveguide.
BUILD_ONLY = {route: set() for route in ROUTES} | {"waveguide": {"memory_budget"}}


def _resolve(table, name, values):
    """Return what a keyword's rule resolves to for these values."""
    resolver = table[name].resolver
    assert resolver is not None, name
    return resolver(values)


def _accepts_none(function, name):
    """Return whether a keyword's annotation admits None."""
    hint = typing.get_type_hints(function)[name]
    return (
        type(None) in typing.get_args(hint)
        or hint is type(None)
        or (isinstance(hint, types.UnionType) and type(None) in hint.__args__)
    )


@pytest.mark.parametrize("route", ROUTES)
def test_the_table_names_every_keyword_of_the_constructor(route):
    signature = inspect.signature(BUILDERS[route])
    # An object route's absorber and scheme are written under its settings.
    skipped = metadata.SETTINGS_KEYWORDS if route in metadata.OBJECT_ROUTES else ()
    expected = [name for name in signature.parameters if name not in skipped]
    assert list(metadata.parameters(route)) == expected
    assert set(skipped) <= set(signature.parameters)


@pytest.mark.parametrize("route", ROUTES)
def test_the_prop_function_adds_only_the_keywords_a_recipe_refuses(route):
    built = set(inspect.signature(BUILDERS[route]).parameters)
    prop = set(inspect.signature(PROPS[route]).parameters)
    assert built - prop == BUILD_ONLY[route]
    assert prop - built <= set(REFUSED_BUILD_KEYS)


@pytest.mark.parametrize("route", ROUTES)
def test_each_default_is_the_signature_s(route):
    signature = inspect.signature(BUILDERS[route])
    for name, entry in metadata.parameters(route).items():
        declared = signature.parameters[name].default
        if declared is inspect.Parameter.empty:
            assert entry.default is metadata.REQUIRED
        else:
            assert entry.default is declared


@pytest.mark.parametrize("route", ROUTES)
def test_the_none_spelling_is_exactly_where_the_default_is_not_none(route):
    builder = BUILDERS[route]
    for name, entry in metadata.parameters(route).items():
        needs = (
            entry.default is not None
            and entry.default is not metadata.REQUIRED
            and _accepts_none(builder, name)
        )
        assert entry.none == needs, name
        # A keyword with the spelling has no "none" of its own to confuse it with.
        own = [form for form in entry.forms if form.choices != ("none",)]
        assert all("none" not in form.choices for form in own), name


def test_the_execution_keys_are_settings_or_the_run_s_budget():
    table = metadata.parameters("execution")
    assert tuple(table) == tuple(EXECUTION_KEYS)
    fields = {field.name for field in SimulationSettings.__dataclass_fields__.values()}
    assert set(table) - fields == {"memory_budget"}
    assert "memory_budget" in inspect.signature(Simulation.run).parameters
    assert table["memory_budget"].none


@pytest.mark.parametrize("route", [*ROUTES, "execution", *REGISTRY])
def test_every_unit_reads_and_every_display_unit_matches_it(route):
    for name, entry in metadata.parameters(route).items():
        units = [Unit(form.unit) for form in entry.forms if form.unit is not None]
        if entry.display is None:
            continue
        assert units, name
        display = Unit(entry.display)
        assert all(unit.dimension == display.dimension for unit in units), name


def _slug(heading):
    """Return MyST's anchor for a heading, as the docs link to it."""
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s+", "-", text)


@pytest.mark.parametrize("route", [*ROUTES, "execution", *REGISTRY])
def test_every_anchor_names_a_page_and_a_heading_of_it(route):
    for name, entry in metadata.parameters(route).items():
        page, _, anchor = entry.anchor.partition("#")
        path = DOCS / page
        assert path.is_file(), (name, page)
        if anchor:
            headings = {
                _slug(line.lstrip("#"))
                for line in path.read_text().splitlines()
                if line.startswith("#")
            }
            assert anchor in headings, (name, entry.anchor)


@pytest.mark.parametrize("route", [*ROUTES, *REGISTRY])
def test_every_entry_has_a_group_level_and_summary(route):
    for name, entry in metadata.parameters(route).items():
        assert entry.group in typing.get_args(metadata.Group), name
        assert entry.level in ("basic", "advanced"), name
        assert entry.summary.endswith("."), name
        assert entry.forms, name


def test_the_resolvable_keywords_are_those_the_sep_measured():
    # SEP 41 section 5.2: these qualify on the keyword routes; the rest stay with the
    # build. The hash test that holds each to that comes with the recorded recipe.
    resolvable = {
        "rtol",
        "atol",
        "dz_max",
        "integrator",
        "stats",
        "temperature",
        "plasma",
        "thg",
        "raman",
        "step_planes",
        "save_planes",
        "loss",
        "model",
    }
    found = {
        name
        for route in ROUTES
        for name, entry in metadata.parameters(route).items()
        if entry.resolvable
    }
    assert found == resolvable
    for name in ("dz_init", "raman_fraction", "absorber", "cladding", "backend"):
        for route in ROUTES:
            entry = metadata.parameters(route).get(name)
            assert entry is None or not entry.resolvable


GNLSE = {
    "wavelength": 835e-9,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 1e-12,
    "width": 100e-15,
    "peak_power": 9.8,
    "shape": "sech",
    "n_save": 3,
    "backend": "numpy_ref",
}
CAPILLARY = {
    "wavelength": 800e-9,
    "wavelength_limits": (400e-9, 2e-6),
    "trange": 200e-15,
    "fwhm": 30e-15,
    "energy": 1e-6,
    "n_save": 2,
    "backend": "numpy_ref",
}


@pytest.mark.parametrize("precision", ["double", "single"])
def test_the_resolvers_say_what_the_build_resolves(precision):
    gnlse = Simulation.from_gnlse(
        0.11, 0.01, [0.0, 0.0, -1.183e-26], **GNLSE, precision=precision
    )
    capillary = Simulation.from_capillary(
        125e-6, 0.1, "He", 1.0, **CAPILLARY, precision=precision
    )
    hollow = [MarcatiliMode(125e-6)]
    waveguide = Simulation.from_waveguide(
        hollow, 0.1, "He", 1.0, **CAPILLARY, precision=precision
    )
    slab = ModeCollection.slab(50e-6, (1,))
    planar = Simulation.from_waveguide(
        slab, 0.1, "Ar", 1.0, **CAPILLARY, precision=precision
    )
    for route, simulation, modes in (
        ("gnlse", gnlse, None),
        ("capillary", capillary, None),
        ("waveguide", waveguide, hollow),
        ("waveguide", planar, slab),
    ):
        table = metadata.parameters(route)
        values = {"precision": precision, "modes": modes}
        settings = simulation.settings
        controller = settings.controller
        assert controller is not None
        assert _resolve(table, "rtol", values).value == controller.rtol
        integrator = _resolve(table, "integrator", values).value
        assert type(integrator) is type(settings.integrator)
        model = _resolve(table, "noise", values).value
        assert model == simulation.noise_resolution.resolved


def test_the_capillary_s_rules_follow_the_envelope_and_the_tdse():
    table = metadata.parameters("capillary")
    assert _resolve(table, "thg", {}).value is True
    assert _resolve(table, "plasma", {"envelope": True}).value is False
    assert _resolve(table, "step_planes", {"tdse": object()}).value == 1
    assert _resolve(table, "pressure", {}).value == 1.0
    assert _resolve(table, "dz_max", {"length": 0.4}).value == 0.2
    assert _resolve(table, "dz_init", {}).value is None
    assert _resolve(table, "raman", {}).value is None
    assert _resolve(table, "absorber_scheme", {}).value == "nonlinear"
    assert "one photon" in _resolve(table, "noise", {}).rule


def test_a_capillary_beside_a_tdse_reads_its_own_defaults_row():
    table = metadata.parameters("capillary")
    resolution = _resolve(table, "rtol", {"tdse": object()})
    assert resolution.value == default_rtol("capillary tdse", "double")
    assert "the capillary tdse route's default" in resolution.rule
    assert "the capillary route's default" in _resolve(table, "rtol", {}).rule


@pytest.mark.parametrize("route", ["gnlse", "waveguide"])
def test_step_planes_saves_the_requested_planes_only_without_a_tdse(route):
    resolution = _resolve(metadata.parameters(route), "step_planes", {})
    assert resolution.value == 0
    assert resolution.rule == "the requested planes only"


def test_a_required_keyword_says_so():
    entry = metadata.parameters("gnlse")["gamma"]
    assert entry.default is metadata.REQUIRED
    assert repr(entry.default) == "REQUIRED"


def test_a_route_without_metadata_is_refused():
    with pytest.raises(DriverError, match="no recipe surface"):
        metadata.parameters("fibre")
    with pytest.raises(DriverError, match="radial, free, planar"):
        metadata.surface("fibre")


@pytest.mark.parametrize("name", REGISTRY)
def test_a_type_s_table_names_every_keyword_its_target_takes(name):
    factory = REGISTRY[name]
    signature = inspect.signature(factory.target)
    keywords = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        and parameter.name not in factory.omitted
    ][len(factory.context) :]
    assert [entry.name for entry in metadata._TYPES[name]] != [] or not keywords
    assert sorted(entry.name for entry in metadata._TYPES[name]) == sorted(keywords)
    assert list(metadata.parameters(name)) == keywords


@pytest.mark.parametrize("name", REGISTRY)
def test_a_type_s_units_are_the_registry_s(name):
    units = REGISTRY[name].units
    for keyword, entry in metadata.parameters(name).items():
        unit = units.get(keyword)
        given = [form.unit for form in entry.forms if form.unit is not None]
        if isinstance(unit, str):
            assert given, keyword  # a quantity's form says its unit
            assert all(Unit(found) == Unit(unit) for found in given), keyword
        elif unit is None:
            assert not given, keyword  # read as it stands: no unit to claim
        else:  # a rule per element, as the phase's
            assert any(form.elements for form in entry.forms), keyword


@pytest.mark.parametrize("name", REGISTRY)
def test_a_type_s_objects_are_registered_types(name):
    for keyword, entry in metadata.parameters(name).items():
        for form in entry.forms:
            if form.kind == "object" and form.recipe:
                assert form.types, keyword
                assert set(form.types) <= set(REGISTRY), keyword


def test_a_type_s_defaults_are_its_target_s():
    pulse = metadata.parameters("GaussianPulse")
    assert pulse["wavelength"].default is metadata.REQUIRED
    assert pulse["m"].default == 1 and pulse["energy"].default is None
    windows = np.asarray(metadata.parameters("StatsSpec")["energy_windows"].default)
    assert windows.shape == (0, 2)  # a dataclass's default factory, called
    assert "grid" not in metadata.parameters("absorber")  # the build context
    assert "tableau" not in metadata.parameters("RungeKutta")  # omitted


def test_a_slab_set_s_energy_is_per_metre_of_y():
    slab = {"modes": {"type": "ModeCollection.slab", "half_gap": 6e-5, "orders": [1]}}
    assert metadata.per_unit_width("waveguide", slab)
    assert not metadata.per_unit_width("capillary", slab)
    assert set(metadata.PER_WIDTH_KEYWORDS) <= set(metadata.parameters("waveguide"))
    assert "per metre of y" in metadata.parameters("waveguide")["energy"].summary
    assert "per metre" not in metadata.parameters("capillary")["peak_power"].summary

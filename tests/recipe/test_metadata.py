# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The parameter metadata agrees with the signatures it describes (SEP 41, S9)."""

import inspect
import re
import types
import typing
from pathlib import Path

import pytest

from solprop.driver import Simulation
from solprop.driver.interface import prop_capillary, prop_gnlse
from solprop.driver.simulation import SimulationSettings
from solprop.exceptions import DriverError
from solprop.recipe import _metadata
from solprop.recipe._format import EXECUTION_KEYS, REFUSED_BUILD_KEYS
from solprop.recipe._quantity import Unit

ROUTES = ("gnlse", "capillary")
DOCS = Path(__file__).resolve().parents[2] / "docs"
BUILDERS = {"gnlse": Simulation.from_gnlse, "capillary": Simulation.from_capillary}
PROPS = {"gnlse": prop_gnlse, "capillary": prop_capillary}


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
    assert list(_metadata.parameters(route)) == list(signature.parameters)


@pytest.mark.parametrize("route", ROUTES)
def test_the_prop_function_adds_only_the_keywords_a_recipe_refuses(route):
    built = set(inspect.signature(BUILDERS[route]).parameters)
    prop = set(inspect.signature(PROPS[route]).parameters)
    assert built <= prop
    assert prop - built <= set(REFUSED_BUILD_KEYS)


@pytest.mark.parametrize("route", ROUTES)
def test_each_default_is_the_signature_s(route):
    signature = inspect.signature(BUILDERS[route])
    for name, entry in _metadata.parameters(route).items():
        declared = signature.parameters[name].default
        if declared is inspect.Parameter.empty:
            assert entry.default is _metadata.REQUIRED
        else:
            assert entry.default is declared


@pytest.mark.parametrize("route", ROUTES)
def test_the_none_spelling_is_exactly_where_the_default_is_not_none(route):
    builder = BUILDERS[route]
    for name, entry in _metadata.parameters(route).items():
        needs = (
            entry.default is not None
            and entry.default is not _metadata.REQUIRED
            and _accepts_none(builder, name)
        )
        assert entry.none == needs, name
        # A keyword with the spelling has no "none" of its own to confuse it with.
        own = [form for form in entry.forms if form.choices != ("none",)]
        assert all("none" not in form.choices for form in own), name


def test_the_execution_keys_are_settings_or_the_run_s_budget():
    table = _metadata.parameters("execution")
    assert tuple(table) == tuple(EXECUTION_KEYS)
    fields = {field.name for field in SimulationSettings.__dataclass_fields__.values()}
    assert set(table) - fields == {"memory_budget"}
    assert "memory_budget" in inspect.signature(Simulation.run).parameters
    assert table["memory_budget"].none


@pytest.mark.parametrize("route", [*ROUTES, "execution"])
def test_every_unit_reads_and_every_display_unit_matches_it(route):
    for name, entry in _metadata.parameters(route).items():
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


@pytest.mark.parametrize("route", [*ROUTES, "execution"])
def test_every_anchor_names_a_page_and_a_heading_of_it(route):
    for name, entry in _metadata.parameters(route).items():
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


@pytest.mark.parametrize("route", ROUTES)
def test_every_entry_has_a_group_level_and_summary(route):
    for name, entry in _metadata.parameters(route).items():
        assert entry.group in typing.get_args(_metadata.Group), name
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
        for name, entry in _metadata.parameters(route).items()
        if entry.resolvable
    }
    assert found == resolvable
    for name in ("dz_init", "raman_fraction", "absorber", "cladding", "backend"):
        for route in ROUTES:
            entry = _metadata.parameters(route).get(name)
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
    for route, simulation in (("gnlse", gnlse), ("capillary", capillary)):
        table = _metadata.parameters(route)
        values = {"precision": precision}
        settings = simulation.settings
        controller = settings.controller
        assert controller is not None
        assert _resolve(table, "rtol", values).value == controller.rtol
        integrator = _resolve(table, "integrator", values).value
        assert type(integrator) is type(settings.integrator)
        model = _resolve(table, "noise", values).value
        assert model == simulation.noise_resolution.resolved


def test_the_capillary_s_rules_follow_the_envelope_and_the_tdse():
    table = _metadata.parameters("capillary")
    assert _resolve(table, "thg", {}).value is True
    assert _resolve(table, "plasma", {"envelope": True}).value is False
    assert _resolve(table, "step_planes", {"tdse": object()}).value == 1
    assert _resolve(table, "pressure", {}).value == 1.0
    assert _resolve(table, "dz_max", {"length": 0.4}).value == 0.2
    assert _resolve(table, "dz_init", {}).value is None
    assert _resolve(table, "raman", {}).value is None
    assert _resolve(table, "absorber_scheme", {}).value == "nonlinear"
    assert "one photon" in _resolve(table, "noise", {}).rule


def test_a_route_without_metadata_is_refused():
    with pytest.raises(DriverError, match="no recipe surface"):
        _metadata.parameters("radial")
    with pytest.raises(DriverError, match="no recipe surface"):
        _metadata.surface("radial")

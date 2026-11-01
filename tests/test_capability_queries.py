# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a route and a species admit, held to the code that refuses (SEP 41, S8)."""

import ast
import inspect
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import solprop
from driver.test_parameter_paths import REFUSALS, _gnlse, _hollow
from driver.test_tdse_entry import _LENGTH, _sae
from solprop import capabilities, exceptions
from solprop.capabilities import RULES, route, species
from solprop.driver import Simulation
from solprop.exceptions import (
    DriverError,
    MaterialError,
    MaterialRangeWarning,
    ParameterPath,
    SolpropError,
    TranslationError,
)
from solprop.modes import CoherentWall
from solprop.modes.antiresonant import TubeLattice
from solprop.physdata import Gas, Mixture
from solprop.physdata import species as built_in
from solprop.physdata.selection import STATUSES
from solprop.recipe.metadata import parameters
from solprop.stepper import LowStorageRungeKutta

ROUTES = ("gnlse", "capillary")
SOURCE = Path(solprop.__file__).parent

# ------------------------------------------------------------------- the routes


@pytest.mark.parametrize("name", ROUTES)
def test_a_route_lists_exactly_its_metadata_keywords(name: str) -> None:
    assert route(name).keywords == tuple(parameters(name))


@pytest.mark.parametrize("name", ROUTES)
def test_a_route_gives_the_choices_of_each_enumerated_keyword(name: str) -> None:
    table = parameters(name)
    expected = {
        keyword: tuple(
            value
            for form in entry.forms
            if form.kind == "choice"
            for value in form.choices
        )
        for keyword, entry in table.items()
    }
    expected = {
        keyword: values
        for keyword, values in expected.items()
        if set(values) - {"none"}
    }
    assert dict(route(name).choices) == expected
    # A keyword whose one choice is the "none" spelling is not enumerated.
    assert "boundary_check" not in route(name).choices


def test_the_capillary_choices_hold_its_named_values() -> None:
    choices = route("capillary").choices
    assert choices["vacuum_end"] == ("choked", "floor")
    assert choices["noise"] == ("default", "input", "modified", "none")


@pytest.mark.parametrize("name", ROUTES)
def test_a_route_holds_the_rules_its_builder_enforces(name: str) -> None:
    statement = route(name)
    assert statement.rules == tuple(
        rule for rule in RULES.values() if name in rule.routes
    )
    for keyword in statement.keywords:
        assert all(keyword in rule.keywords for rule in statement.rules_for(keyword))
    assert statement.rules_for("not-a-keyword") == ()


def test_the_tdse_rules_are_the_capillary_route_s_alone() -> None:
    assert "tdse-helium" in {rule.id for rule in route("capillary").rules}
    assert "tdse-helium" not in {rule.id for rule in route("gnlse").rules}
    assert RULES["tdse-helium"].species_condition == "helium alone"


@pytest.mark.parametrize("rule", RULES.values(), ids=list(RULES))
def test_every_rule_names_keywords_of_its_routes(rule: capabilities.Rule) -> None:
    assert rule.routes
    assert set(rule.routes) <= set(ROUTES)
    assert rule.reason.endswith(".")
    execution = set(parameters("execution"))
    for name in rule.routes:
        assert set(rule.keywords) <= set(parameters(name)) | execution, name


def test_rule_ids_are_short_kebab_case() -> None:
    for identifier in RULES:
        assert identifier == identifier.lower()
        assert identifier.replace("-", "").isalnum()
        assert RULES[identifier].id == identifier


@pytest.mark.parametrize("name", ["execution", "free", "Capillary"])
def test_a_route_without_metadata_is_refused_with_the_routes(name: str) -> None:
    with pytest.raises(DriverError, match="gnlse, capillary") as caught:
        route(name)
    assert caught.value.parameter == "route"


# -------------------------------------------------------- the rules in the source


def _exception_names() -> set[str]:
    return {
        name
        for name, value in inspect.getmembers(exceptions, inspect.isclass)
        if issubclass(value, SolpropError)
    }


def _marked_sites() -> Iterator[tuple[str, ast.Call, ast.keyword]]:
    """Yield every exception constructed with ``rule=`` in the package's source."""
    names = _exception_names()
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else None
            called = func.id if isinstance(func, ast.Name) else called
            if called not in names:
                continue
            for keyword in node.keywords:
                if keyword.arg == "rule":
                    where = f"{path.relative_to(SOURCE)}:{node.lineno}"
                    yield where, node, keyword


def _literal_paths(node: ast.Call) -> list[str]:
    """Return the literal paths a call's ``parameter`` names, where it is literal."""
    for keyword in node.keywords:
        if keyword.arg != "parameter":
            continue
        value = keyword.value
        elements = value.elts if isinstance(value, ast.Tuple) else [value]
        return [
            element.value
            for element in elements
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
    return []


def _head(path: str) -> str:
    """Return the keyword a parameter path starts from (``wall.method`` -> wall)."""
    return path.split(".", 1)[0].split("[", 1)[0]


def test_every_marked_refusal_names_a_listed_rule() -> None:
    sites = list(_marked_sites())
    assert sites, "no refusal is marked with a rule"
    used: set[str] = set()
    for where, node, keyword in sites:
        value = keyword.value
        assert isinstance(value, ast.Constant), f"{where}: the rule is not a literal"
        assert isinstance(value.value, str), f"{where}: the rule is not a string"
        assert value.value in RULES, f"{where}: {value.value!r} is not in RULES"
        used.add(value.value)
        rule = RULES[value.value]
        for path in _literal_paths(node):
            assert _head(path) in rule.keywords, f"{where}: {path} is not in {rule.id}"
    assert set(RULES) - used == set(), "listed rules that no refusal enforces"


# -------------------------------------------------------- the rules when broken

# The rule each refusal of the parameter-path cases enforces; a case not named here
# is refused for a value, and carries no rule.
RULE_OF_CASE = {
    "gnlse-no-window": "grid-or-limits",
    "gnlse-no-grid": "grid-or-limits",
    "gnlse-field-grid": "gnlse-envelope-grid",
    "gnlse-dispersion-twice": "betas-beta-or-fibre",
    "gnlse-gamma-and-n2": "gamma-or-n2",
    "gnlse-fibre-loss": "fibre-carries-loss",
    "gnlse-gaussian-width": "gauss-fwhm",
    "gnlse-pulses-beside": "pulses-or-pulse-keywords",
    "gnlse-planes-twice": "save-planes-or-n-save",
    "gnlse-absorber": "absorber-scheme-needs-absorber",
    "gnlse-seed": "seed-needs-noise",
    "gnlse-pulse-duration": "fwhm-or-width",
    "gnlse-pulse-scale": "energy-or-peak-power",
    "capillary-raman-kerr": "raman-needs-kerr",
    "capillary-polarisation": "scalar-capillary-polarisation",
    "capillary-averaged-modes": "mode-average-one-mode",
    "capillary-averaged-rule": "mode-average-no-quadrature",
    "capillary-envelope-thg": "envelope-plasma-no-thg",
    "capillary-integration-off": "plasma-integration-needs-plasma",
    "capillary-no-nonlinearity": "capillary-needs-nonlinearity",
    "capillary-grid-twice": "grid-or-limits",
    "capillary-no-limits": "grid-or-limits",
    "capillary-mixture-pressure": "mixture-carries-pressure",
    "capillary-wall-taper": "wall-keeps-taper-coupling",
    "capillary-wall-loss": "wall-is-the-loss",
    "capillary-tdse-execution": "tdse-execution-needs-tdse",
    "capillary-tdse-plasma": "tdse-owns-electronic",
    "capillary-raman-atom": "raman-needs-molecule",
}


def _paths(parameter: ParameterPath | None) -> tuple[str, ...]:
    if parameter is None:
        return ()
    return (parameter,) if isinstance(parameter, str) else tuple(parameter)


def _assert_enforces(error: SolpropError, rule: str | None) -> None:
    assert error.rule == rule
    if rule is not None:
        assert {_head(path) for path in _paths(error.parameter)} <= set(
            RULES[rule].keywords
        )


def test_every_case_maps_to_a_known_case() -> None:
    assert set(RULE_OF_CASE) <= {case.id for case in REFUSALS}


@pytest.mark.parametrize(("build", "kind", "parameter", "match"), REFUSALS)
def test_a_refusal_carries_the_rule_it_enforces(
    request: pytest.FixtureRequest,
    build: Callable[[], Simulation],
    kind: type[SolpropError],
    parameter: ParameterPath,
    match: str,
) -> None:
    case = request.node.callspec.id
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        with pytest.raises(kind, match=match) as caught:
            build()
    assert caught.value.parameter == parameter
    _assert_enforces(caught.value, RULE_OF_CASE.get(case))


def _argon_tdse() -> None:
    values: dict[str, Any] = {
        "wavelength": 800e-9,
        "wavelength_limits": (400e-9, 1.6e-6),
        "trange": 60e-15,
        "fwhm": 6e-15,
        "energy": 1e-12,
        "modes": 1,
        "n_r": 4,
        "noise": None,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        Simulation.from_capillary(
            256e-6, _LENGTH, "Ar", 0.1, tdse=_sae(), **values
        ).translate()


BROKEN = [
    pytest.param(
        lambda: _gnlse(integrator=LowStorageRungeKutta()),
        DriverError,
        ("integrator", "schedule"),
        "low-storage-needs-schedule",
        id="low-storage-needs-schedule",
    ),
    pytest.param(
        lambda: _hollow(lattice=TubeLattice(180e-9, 10.38e-6, 6), model="reduced"),
        DriverError,
        ("lattice", "model"),
        "lattice-full-model",
        id="lattice-full-model",
    ),
    pytest.param(
        lambda: _hollow(wall=CoherentWall(), model="reduced"),
        DriverError,
        ("wall", "model"),
        "wall-full-model",
        id="wall-full-model",
    ),
    pytest.param(
        lambda: _hollow(
            100e-6,
            0.01,
            Mixture.fill({"He": 1.0}, 1.0),
            None,
            susceptibility=built_in("He").susceptibility,
        ),
        DriverError,
        ("gas", "susceptibility"),
        "mixture-carries-susceptibility",
        id="mixture-carries-susceptibility",
    ),
    pytest.param(
        _argon_tdse,
        TranslationError,
        ("tdse", "gas"),
        "tdse-helium",
        id="tdse-helium",
    ),
]


@pytest.mark.parametrize(("build", "kind", "parameter", "rule"), BROKEN)
def test_breaking_a_rule_raises_with_its_id_and_keywords(
    build: Callable[[], object],
    kind: type[SolpropError],
    parameter: ParameterPath,
    rule: str,
) -> None:
    with pytest.raises(kind) as caught:
        build()
    assert caught.value.parameter == parameter
    assert caught.value.rule == rule
    assert set(_paths(parameter)) == set(RULES[rule].keywords)


# ------------------------------------------------------------------ the species


def test_helium_is_an_atom_with_no_raman_response() -> None:
    helium = species("He")
    assert helium.name == "He"
    assert not helium.molecule
    assert helium.raman == "atom"
    assert not helium.raman_by_default
    assert helium.ionisation[0].name == "recommended"
    assert helium.ionisation[0].status == "validated"
    assert {model.name for model in helium.ionisation} >= {"ppt_ilkov", "one_electron"}


def test_nitrogen_has_a_raman_response_on_by_default() -> None:
    nitrogen = species(Gas.N2)
    assert nitrogen.molecule
    assert nitrogen.raman == "validated"
    assert nitrogen.raman_by_default
    assert "moadk_zhao_2010" in {model.name for model in nitrogen.ionisation}


def test_a_provisional_raman_response_is_reported_and_off_by_default() -> None:
    nitrous_oxide = species("N2O")
    assert nitrous_oxide.raman == "provisional"
    assert not nitrous_oxide.raman_by_default


def test_an_untested_ionisation_model_is_reported_as_optmat_has_it() -> None:
    recommended = species("SF6").ionisation[0]
    assert recommended.name == "recommended"
    assert recommended.status == "untested"
    assert recommended.reason


@pytest.mark.parametrize("gas", [member for member in Gas if member is not Gas.Air])
def test_the_species_facts_are_those_a_fill_uses(gas: Gas) -> None:
    statement = species(gas)
    material = built_in(gas)
    assert statement.wavelength_range == material.susceptibility.wavelength_range
    low, high = statement.temperature_range
    assert 0.0 < low < high
    assert statement.records == material.material_records
    assert statement.ionisation, "every built-in gas but air has an optmat rate"
    assert all(model.status in STATUSES for model in statement.ionisation)
    assert all(record.status in STATUSES for record in statement.records)


def test_a_gas_neither_rate_family_covers_has_no_ionisation_model() -> None:
    assert capabilities._ionisation_models(Gas.Air) == ()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("name", ["air", "Air", Gas.Air])
def test_air_is_refused_naming_its_constituents(name: str | Gas) -> None:
    with pytest.raises(DriverError, match="N2 and O2") as caught:
        species(name)
    assert caught.value.parameter == "gas"


@pytest.mark.parametrize("name", ["Xx", "n2", ""])
def test_an_unknown_species_is_refused(name: str) -> None:
    with pytest.raises(MaterialError, match="unknown gas") as caught:
        species(name)
    assert caught.value.parameter == "gas"

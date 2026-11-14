# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recipes: read, check, build, write and export (SEP 41, S1 and S10)."""

import hashlib
import json
import signal
import subprocess
import sys
import textwrap
import warnings
from typing import Any, cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import TypeCheckError

from builders import HASH_PROBE, run_python
from runtime_types import TYPECHECKED
from solprop.cli import main
from solprop.driver import Simulation
from solprop.exceptions import BackendError, DriverError, MaterialRangeWarning
from solprop.profiles import PiecewiseLinear
from solprop.pulses import DataPulse, SechPulse
from solprop.recipe import Recipe, RecipeError, RecipeIssue, parameters
from solprop.recipe.core import ROUTES
from solprop.recipe.document import FORMAT, parse_text
from solprop.recipe.quantity import Unit, format_quantity, parse_quantity
from solprop.recipe.registry import (
    REGISTRY,
    BuildContext,
    build_object,
    coerce,
    decode,
    decode_keyword,
    unit_of,
)
from solprop.stepper import Multistep

# A GNLSE that builds in milliseconds: the soliton of the GNLSE guide's example.
GNLSE = textwrap.dedent(
    """\
    # A comment kept by a verbatim save.
    [recipe]
    format = 1
    route = "gnlse"
    title = "soliton"

    [build]
    gamma = 0.11
    length = "1 cm"
    betas = [0, 0, "-0.01183 ps^2/m"]
    wavelength = "835 nm"
    wavelength_limits = ["500 nm", "2 um"]
    trange = "4 ps"
    width = "100 fs"
    peak_power = "9.8 W"
    shape = "sech"
    n_save = 3
    noise = "none"
    backend = "numpy_ref"
    """
)
GNLSE_CALL = {
    "gamma": 0.11,
    "length": 0.01,
    "betas": [0.0, 0.0, -1.183e-26],
    "wavelength": 835e-9,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 4e-12,
    "width": 100e-15,
    "peak_power": 9.8,
    "shape": "sech",
    "n_save": 3,
    "noise": None,
    "backend": "numpy_ref",
}


def _hash(simulation):
    return simulation.metadata().configuration_hash()


@pytest.fixture
def recipe():
    return Recipe.loads(GNLSE)


def test_a_recipe_builds_the_call_it_stands_for(recipe):
    assert _hash(recipe.build()) == _hash(Simulation.from_gnlse(**GNLSE_CALL))


def test_an_unedited_recipe_saves_verbatim_and_an_edited_one_canonically(
    recipe, tmp_path
):
    path = tmp_path / "soliton.toml"
    recipe.save(path)
    assert path.read_text() == GNLSE
    recipe.set("length", 0.02)
    recipe.save(path)
    text = path.read_text()
    assert "# A comment" not in text
    # The canonical text spells each quantity in its display unit.
    assert 'length = "0.02 m"' in text
    assert 'wavelength = "835 nm"' in text
    again = Recipe.load(path)
    assert again == recipe
    assert _hash(again.build()) == _hash(
        Simulation.from_gnlse(**{**GNLSE_CALL, "length": 0.02})
    )


def test_recipes_compare_by_value_not_text(recipe):
    spaced = Recipe.loads(GNLSE.replace('length = "1 cm"', 'length   =   "1 cm"'))
    assert spaced == recipe
    assert Recipe.loads(GNLSE.replace('"1 cm"', "0.01")) == recipe
    assert recipe != Recipe.loads(GNLSE.replace('"1 cm"', '"2 cm"'))
    assert recipe.__eq__(object()) is NotImplemented
    assert repr(recipe) == "Recipe(gnlse, 'soliton')"


def test_get_and_set_take_paths_and_check_the_value(recipe):
    assert recipe.get("betas[2]") == "-0.01183 ps^2/m"
    recipe.set("betas[2]", "-0.02 ps^2/m")
    assert recipe.get("betas[2]") == "-0.02 ps^2/m"
    with pytest.raises(RecipeError, match="length") as caught:
        recipe.set("length", "3 bar")
    assert caught.value.parameter == "length"
    with pytest.raises(RecipeError, match=r"lengthx"):
        recipe.set("lengthx", 1.0)


def test_a_recipe_naming_the_retired_mlx_backend_is_refused_by_name():
    """SEP 47 retired mlx: its recipe is refused, never run on another backend."""
    retired = Recipe.loads(GNLSE.replace('backend = "numpy_ref"', 'backend = "mlx"'))
    with pytest.raises(BackendError, match="'mlx' is not a backend"):
        retired.build()


@pytest.mark.parametrize(
    ("old", "new", "parameter", "reason"),
    [
        (
            'trange = "300 fs"',
            'trange = "300 fs", transforms = "fftw"',
            "grid.transforms",
            "transforms was retired in SEP 47",
        ),
        (
            'type = "CoherentWall"',
            'type = "CoherentWall", method = "cf4"',
            "wall.method",
            "CF4 coupled-flow sections were retired",
        ),
        (
            'type = "CoherentWall"',
            'type = "CoherentWall", growth_limit = 2.0',
            "wall.growth_limit",
            "the default frozen-residual flow is used",
        ),
        (
            'type = "CoherentWall"',
            'type = "CoherentWall", fallback = "frozen_residual"',
            "wall.fallback",
            "remove the key",
        ),
    ],
)
def test_a_retired_object_keyword_is_refused_with_its_reason(
    old, new, parameter, reason
):
    """A retired keyword names why it went and the fix, not an unknown key."""
    text = Recipe.example("coherent_wall").to_text()
    assert old in text
    (issue,) = Recipe.loads(text.replace(old, new, 1)).check()
    assert issue.parameter == parameter
    assert reason in issue.message
    assert "unknown key" not in issue.message


def test_a_double_precision_recipe_still_loads_and_single_is_refused():
    """``precision = "double"`` is what solprop does anyway, so it is dropped."""
    double = Recipe.loads(GNLSE.replace("[build]\n", '[build]\nprecision = "double"\n'))
    with pytest.raises(RecipeError):
        double.get("precision")
    assert _hash(double.build()) == _hash(Recipe.loads(GNLSE).build())
    with pytest.raises(RecipeError, match=r"binary64 only \(ADR-077\); remove the key"):
        Recipe.loads(GNLSE.replace("[build]\n", '[build]\nprecision = "single"\n'))


def test_unknown_keys_and_routes_are_refused():
    with pytest.raises(RecipeError, match="wavelengthx"):
        Recipe.loads(GNLSE.replace("wavelength =", "wavelengthx ="))
    with pytest.raises(RecipeError, match="the routes a recipe describes"):
        Recipe.loads(GNLSE.replace('route = "gnlse"', 'route = "fibre"'))
    with pytest.raises(RecipeError, match="output"):
        Recipe.loads(GNLSE + 'output = "run.h5"\n')


def test_from_keywords_takes_plain_values_and_refuses_objects():
    recipe = Recipe.from_keywords("gnlse", title="soliton", **GNLSE_CALL)
    assert _hash(recipe.build()) == _hash(Simulation.from_gnlse(**GNLSE_CALL))
    assert recipe.title == "soliton"
    with pytest.raises(RecipeError, match="a Multistep cannot be written"):
        Recipe.from_keywords("gnlse", **GNLSE_CALL, integrator=Multistep())
    with pytest.raises(RecipeError, match="a callable"):
        Recipe.from_keywords("gnlse", **GNLSE_CALL, beta=lambda omega: omega)
    complex_value = Recipe.from_keywords("capillary", cladding=1.45 + 0.01j)
    assert complex_value.get("cladding") == {"re": 1.45, "im": 0.01}
    absorber = {"type": "absorber", "boundary_N": 30}
    table = Recipe.from_keywords("gnlse", **GNLSE_CALL, absorber=absorber)
    assert table.get("absorber") == absorber
    written = Recipe.loads(GNLSE + 'absorber = {type = "absorber", boundary_N = 30}\n')
    assert _hash(table.build()) == _hash(written.build())


def test_execution_settings_reach_the_settings_and_the_run(recipe):
    text = GNLSE + '\n[execution]\nfields = "planes"\nfield_planes = [0, -1]\n'
    simulation = Recipe.loads(text).build()
    assert simulation.settings.fields == "planes"
    assert simulation.settings.field_planes == (0, -1)
    for source, built in (
        (Recipe.loads(text), simulation),
        (recipe, recipe.build()),
    ):
        metadata = source.run().metadata
        assert metadata is not None
        assert metadata.configuration_hash() == _hash(built)


@pytest.mark.parametrize("value", ["none", "true", 1])
def test_a_resume_history_that_is_not_a_boolean_is_refused(value):
    # Issue #703: a truthy string such as "none" was read by the history policy as
    # True, the opposite of what it says, by the build and the Python export alike.
    document = Recipe.loads(GNLSE).to_dict()
    document["execution"] = {"resume_history": value}
    recipe = Recipe.from_dict(document)
    if TYPECHECKED:
        # The annotation refuses it first when the runtime checks are on.
        with pytest.raises(TypeCheckError):
            recipe.build()
        return
    with pytest.raises(DriverError, match="resume_history must be True, False or None"):
        recipe.build()
    assert [issue.parameter for issue in recipe.check()] == ["resume_history"]


def test_a_registered_object_is_built_from_its_table():
    text = GNLSE.replace(
        'width = "100 fs"\npeak_power = "9.8 W"\nshape = "sech"\n',
        '[[build.pulses]]\ntype = "SechPulse"\nwavelength = "835 nm"\n'
        'width = "100 fs"\npeak_power = "9.8 W"\n',
    )
    # The pulses table follows the build keywords, as TOML requires.
    text = text.replace('n_save = 3\nnoise = "none"\nbackend = "numpy_ref"\n', "")
    text = text.replace(
        "[[build.pulses]]",
        'n_save = 3\nnoise = "none"\nbackend = "numpy_ref"\n\n[[build.pulses]]',
    )
    recipe = Recipe.loads(text)
    call = {
        key: value
        for key, value in GNLSE_CALL.items()
        if key not in ("width", "peak_power", "shape")
    }
    pulse = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8)
    assert _hash(recipe.build()) == _hash(Simulation.from_gnlse(**call, pulses=[pulse]))


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (
            {
                "type": "GaussianPulse",
                "wavelength": "800 nm",
                "fwhm": "30 fs",
                "energy": "1 uJ",
            },
            "GaussianPulse",
        ),
        (
            {
                "type": "PressureGradient.two_point",
                "length": "1 m",
                "start_bar": 1.2,
                "end_bar": "0 bar",
            },
            "PressureGradient",
        ),
        (
            {
                "type": "Mixture.fill",
                "fractions": {"He": 0.5, "Ar": 0.5},
                "pressure": "1 bar",
            },
            "Mixture",
        ),
        (
            {
                "type": "EnvelopeGrid.from_limits",
                "length": "1 m",
                "reference_wavelength": "800 nm",
                "wavelength_limits": ["400 nm", "2 um"],
                "trange": "200 fs",
            },
            "EnvelopeGrid",
        ),
        ({"type": "StatsSpec", "names": ["energy"]}, "StatsSpec"),
        ({"type": "BoundaryCheck", "budget": 1e-4}, "BoundaryCheck"),
        (
            {
                "type": "FixedSchedule.uniform",
                "z_start": 0,
                "z_end": "1 cm",
                "dz": "1 mm",
            },
            "FixedSchedule",
        ),
        ({"type": "Multistep", "order": 4}, "Multistep"),
        ({"type": "RungeKutta"}, "RungeKutta"),
        ({"type": "LowStorageRungeKutta"}, "LowStorageRungeKutta"),
        (
            {
                "type": "TubeLattice",
                "wall_thickness": "0.5 um",
                "tube_radius": "20 um",
                "n_tubes": 6,
            },
            "TubeLattice",
        ),
        ({"type": "CoherentWall"}, "CoherentWall"),
        (
            {"type": "MaterialSelection", "electronic": "recommended"},
            "MaterialSelection",
        ),
        ({"type": "Constant", "value": 1.0}, "Constant"),
        ({"type": "Linear", "start": 1.0, "end": 0.0, "length": "1 m"}, "Linear"),
        ({"type": "Mixture.air", "pressure": 1.0}, "Mixture"),
        ({"type": "Mixture.partial_pressures", "pressures": {"He": 1.0}}, "Mixture"),
        (
            {"type": "MarcatiliMode", "radius": "125 um", "kind": "TE", "n": 0},
            "MarcatiliMode",
        ),
        (
            {
                "type": "RectMode",
                "half_width_x": "50 um",
                "half_width_y": "10 um",
                "polarisation": "y",
                "cladding": {"re": 1.45, "im": 0.0},
            },
            "RectMode",
        ),
        ({"type": "SlabMode", "half_gap": "63.5 um", "order": 3}, "SlabMode"),
        (
            {"type": "ModeCollection.slab", "half_gap": "63.5 um", "orders": [1, 3]},
            "ModeCollection",
        ),
    ],
)
def test_each_registered_type_builds_from_a_table(table, expected):
    built = build_object(table, BuildContext(), "value")
    assert type(built).__name__ == expected


def test_a_type_outside_the_registry_or_a_keyword_outside_its_factory_is_refused():
    with pytest.raises(RecipeError, match="CustomPulse"):
        build_object({"type": "CustomPulse"}, BuildContext(), "pulses[0]")
    with pytest.raises(RecipeError, match="colour") as caught:
        build_object({"type": "BoundaryCheck", "colour": 1}, BuildContext(), "x")
    assert caught.value.parameter == "x"
    with pytest.raises(RecipeError, match="GaussianPulse"):
        build_object({"type": "GaussianPulse"}, BuildContext(), "pulses[0]")


def test_a_factory_on_the_grid_waits_for_the_route_s_grid():
    with pytest.raises(RecipeError, match="not built yet"):
        build_object({"type": "absorber"}, BuildContext(), "absorber")
    text = GNLSE + 'absorber = {type = "absorber", boundary_N = 30}\n'
    recipe = Recipe.loads(text)
    built = recipe.build()
    assert built.settings.absorber is not None
    assert recipe.check() == []
    with pytest.raises(RecipeError, match="give grid"):
        recipe.to_python()


def test_data_files_are_read_checked_and_confined(tmp_path):
    values = np.array([0.0, 0.0, -1.183e-26])
    np.save(tmp_path / "betas.npy", values)
    digest = hashlib.sha256((tmp_path / "betas.npy").read_bytes()).hexdigest()
    text = GNLSE.replace(
        'betas = [0, 0, "-0.01183 ps^2/m"]',
        f'betas = {{file = "betas.npy", sha256 = "{digest}"}}',
    )
    (tmp_path / "soliton.toml").write_text(text)
    recipe = Recipe.load(tmp_path / "soliton.toml")
    assert _hash(recipe.build()) == _hash(Simulation.from_gnlse(**GNLSE_CALL))
    with pytest.raises(RecipeError, match="read from text"):
        Recipe.loads(text).build()
    (tmp_path / "bad.toml").write_text(text.replace(digest, "0" * 64))
    with pytest.raises(RecipeError):
        Recipe.load(tmp_path / "bad.toml").build()


def test_a_measured_spectrum_is_a_data_file_entry(tmp_path):
    frequency = np.linspace(330e12, 420e12, 200)
    power = np.exp(-(((frequency - 375e12) / 10e12) ** 2))
    np.savetxt(
        tmp_path / "spectrum.csv", np.column_stack([frequency, power]), delimiter=","
    )
    digest = hashlib.sha256((tmp_path / "spectrum.csv").read_bytes()).hexdigest()
    table = {
        "type": "load_spectrum",
        "file": "spectrum.csv",
        "sha256": digest,
        "energy": "1 uJ",
    }
    pulse = build_object(table, BuildContext(base=tmp_path), "pulses[0]")
    assert isinstance(pulse, DataPulse)
    assert pulse.energy == 1e-6
    with pytest.raises(RecipeError, match="read from text"):
        build_object(table, BuildContext(), "pulses[0]")


def test_decode_reads_quantities_complex_numbers_mappings_and_pairs():
    context = BuildContext()
    assert decode("800 nm", "m", context, "wavelength") == 800e-9
    assert decode({"re": 1.0, "im": -2.0}, None, context, "x") == 1 - 2j
    assert decode({"He": 0.5}, None, context, "gas") == {"He": 0.5}
    table = parameters("capillary")
    pairs = decode_keyword(
        table["pressure"], [["0 m", "1 m"], ["1 bar", "0 bar"]], context
    )
    assert pairs == [[0.0, 1.0], [1.0, 0.0]]
    with pytest.raises(RecipeError, match="10000"):
        decode(list(range(10_001)), None, context, "save_planes")
    with pytest.raises(RecipeError, match="Python export"):
        decode_keyword(table["tdse"], {"type": "x"}, context)
    # A value that is neither a number nor a string is passed on as it stands.
    assert decode(None, "m", context, "x") is None


def test_a_list_reaches_an_array_parameter_as_an_array():
    positions = coerce(PiecewiseLinear, "positions", [0.0, 0.1])
    assert isinstance(positions, np.ndarray)
    assert positions.dtype == np.float64
    assert coerce(PiecewiseLinear, "positions", 0.1) == 0.1
    profile = build_object(
        {"type": "PiecewiseLinear", "positions": ["0 mm", "10 mm"], "values": [1, 0]},
        BuildContext(),
        "pressure",
    )
    assert isinstance(profile, PiecewiseLinear)
    np.testing.assert_array_equal(profile.positions, [0.0, 0.01])


def test_a_two_row_list_finds_no_form_of_a_one_row_keyword():
    betas = parameters("gnlse")["betas"]
    assert callable(unit_of(betas, [0, 0, "-0.01183 ps^2/m"]))
    assert unit_of(betas, [[0, 0, "-0.01183 ps^2/m"]]) is None
    pressure = parameters("capillary")["pressure"]
    assert callable(unit_of(pressure, [["0 m", "1 m"], ["1 bar", "0 bar"]]))
    assert unit_of(pressure, ["1 bar", "0 bar"]) == "bar"


@pytest.mark.parametrize(
    ("name", "key"),
    [("low_call_stepping", "pressure"), ("coherent_wall_taper", "radius")],
)
def test_a_canonical_save_keeps_rows_and_tables_as_written(name, key, tmp_path):
    # The (z, p) rows take a unit each, which no display unit spells; a profile is a
    # table of its own.
    recipe = Recipe.example(name)
    authored = recipe.get(key)
    recipe.set("length", recipe.get("length"))  # edited: saved canonically
    path = tmp_path / "edited.toml"
    recipe.save(path)
    again = Recipe.load(path)
    assert again.get(key) == authored
    assert [issue for issue in again.check() if issue.severity == "error"] == []
    assert again == recipe


def test_a_none_a_keyword_does_not_take_is_written_for_check_to_name():
    document = Recipe.loads(GNLSE).to_dict()
    document["build"]["fwhm"] = "none"
    recipe = Recipe.from_dict(document)
    assert 'fwhm = "none"' in recipe.to_text()
    (issue,) = recipe.check()
    assert issue.parameter == "fwhm"
    assert "'none' is not a quantity" in issue.message


def test_check_names_rows_given_to_a_one_row_keyword():
    rows = 'betas = [[0, 0, "-0.01183 ps^2/m"]]'
    bad = Recipe.loads(GNLSE.replace('betas = [0, 0, "-0.01183 ps^2/m"]', rows))
    (issue,) = bad.check()
    assert issue.parameter == "betas"
    assert "one row of numbers" in issue.message


def test_check_names_the_keyword_of_a_table_its_constructor_refuses():
    document = Recipe.example("cross_phase_modulation").to_dict()
    first = cast("list[dict[str, object]]", document["build"]["pulses"])[0]
    first.pop("energy", None)
    first.pop("peak_power", None)
    (issue,) = Recipe.from_dict(document).check()
    assert issue.parameter == "pulses"
    assert issue.severity == "error"


def test_check_reports_each_value_at_its_path(recipe):
    bad = Recipe.loads(GNLSE.replace('length = "1 cm"', 'length = "1 bar"'))
    issues = bad.check()
    assert [issue.parameter for issue in issues] == ["length"]
    assert issues[0].severity == "error"
    missing = Recipe.loads(GNLSE.replace("gamma = 0.11\n", ""))
    assert RecipeIssue("gamma is required", "gamma", "error") in missing.check()
    assert recipe.check() == []


@pytest.mark.parametrize(
    "call",
    [
        Recipe.build,
        Recipe.resolve,
        Recipe.modes,
        Recipe.preview_input,
        lambda recipe: recipe.run(None),
    ],
    ids=["build", "resolve", "modes", "preview_input", "run"],
)
def test_a_missing_required_keyword_is_refused_before_the_route(call):
    missing = Recipe.loads(GNLSE.replace('wavelength = "835 nm"\n', ""))
    with pytest.raises(
        RecipeError, match=r"^wavelength: a required keyword is missing$"
    ) as caught:
        call(missing)
    assert caught.value.parameter == "wavelength"


def test_several_missing_keywords_are_all_named_with_the_first_as_parameter():
    text = GNLSE.replace("gamma = 0.11\n", "").replace('wavelength = "835 nm"\n', "")
    with pytest.raises(
        RecipeError, match="the required keywords gamma, wavelength are missing"
    ) as caught:
        Recipe.loads(text).build()
    assert caught.value.parameter == "gamma"


def test_check_reports_a_refusal_of_the_build_with_its_parameter():
    text = GNLSE.replace("n_save = 3\n", 'n_save = 3\nsave_planes = ["5 mm"]\n')
    issues = Recipe.loads(text).check()
    assert len(issues) == 1
    assert issues[0].severity == "error"


def test_check_reports_warnings_too():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        recipe = Recipe.example("rdw")
        recipe.set("length", "1 cm")
        issues = recipe.check()
    assert all(issue.severity == "warning" for issue in issues)


def test_the_examples_are_listed_and_loaded():
    assert "rdw" in Recipe.examples()
    assert Recipe.example("rdw").route == "capillary"
    with pytest.raises(RecipeError, match="rdx"):
        Recipe.example("rdx")


def test_the_export_is_a_script_that_builds_the_same_run(recipe, tmp_path):
    text = GNLSE + '\n[execution]\nfields = "planes"\nfield_planes = [0, -1]\n'
    for source, form in ((recipe, "prop_gnlse("), (Recipe.loads(text), "from_gnlse(")):
        script = source.to_python()
        assert form in script
        assert "# 835 nm" in script
        probe = textwrap.dedent(
            """\
            import json
            from solprop.driver.simulation import Simulation

            def run(self, *args, **kwargs):
                settings = self.settings
                print(json.dumps([self.metadata().configuration_hash(),
                                  settings.fields, list(settings.field_planes)]))
                raise SystemExit(0)

            Simulation.run = run
            """
        )
        path = tmp_path / "export.py"
        path.write_text(probe + script)
        result = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, check=True
        )
        built = source.build()
        assert json.loads(result.stdout) == [
            _hash(built),
            built.settings.fields,
            list(built.settings.field_planes),
        ]
    assert "output='run.h5'" in recipe.to_python("run.h5")


def _data_file(path, data):
    path.write_bytes(data)
    return {"file": path.name, "sha256": hashlib.sha256(data).hexdigest()}


def _exported_recipe(kind, tmp_path):
    if kind == "data files":
        # An array and a measured spectrum read from files beside the recipe.
        values = tmp_path / "betas.npy"
        np.save(values, np.array([0.0, 0.0, -1.183e-26]))
        frequency = np.linspace(315e12, 405e12, 200)
        power = np.exp(-(((frequency - 359e12) / 10e12) ** 2))
        np.savetxt(
            tmp_path / "spectrum.csv",
            np.column_stack([frequency, power]),
            delimiter=",",
        )
        document = Recipe.loads(GNLSE).to_dict()
        build = document["build"]
        build["betas"] = _data_file(values, values.read_bytes())
        for key in ("width", "peak_power", "shape"):
            del build[key]
        spectrum = tmp_path / "spectrum.csv"
        build["pulses"] = [
            {
                "type": "load_spectrum",
                **_data_file(spectrum, spectrum.read_bytes()),
                "energy": "1 pJ",
            }
        ]
        return Recipe(document, base=tmp_path)
    # A fill given as a plain table of fractions.
    recipe = Recipe.example("overview")
    recipe.set("gas", {"He": 0.5, "Ne": 0.5})
    return recipe


@pytest.mark.child_runs_auto
@pytest.mark.parametrize("kind", ["data files", "a plain table"])
def test_the_export_writes_data_files_and_plain_tables(kind, tmp_path):
    recipe = _exported_recipe(kind, tmp_path)
    script = recipe.to_python()
    if kind == "data files":
        assert "numpy.load('betas.npy', allow_pickle=False)" in script
        assert "load_spectrum('spectrum.csv'," in script
    else:
        assert "gas={'He': 0.5, 'Ne': 0.5}" in script
    path = tmp_path / "export.py"
    path.write_text(HASH_PROBE + script)
    result = run_python(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        built = recipe.build()
    assert result.stdout.split() == [_hash(built)]


def test_the_command_line_checks_exports_and_runs(tmp_path, capsys):
    path = tmp_path / "soliton.toml"
    path.write_text(GNLSE)
    assert main(["check", str(path)]) == 0
    assert "builds" in capsys.readouterr().out
    assert main(["export", str(path), "--python"]) == 0
    assert "prop_gnlse(" in capsys.readouterr().out
    assert (
        main(["run", str(path), "-o", str(tmp_path / "out.h5"), "--progress", "none"])
        == 0
    )
    assert (tmp_path / "out.h5").is_file()
    bad = tmp_path / "bad.toml"
    bad.write_text(GNLSE.replace('"1 cm"', '"1 bar"'))
    assert main(["check", str(bad)]) == 1
    assert "error: length:" in capsys.readouterr().out
    assert main(["check", str(tmp_path / "missing.toml")]) == 2


def test_the_command_line_forwards_scan(monkeypatch):
    calls = []
    monkeypatch.setattr("solprop.scan.main", lambda argv: calls.append(argv) or 0)
    assert main(["scan", "script.py", "--local"]) == 0
    assert calls == [["script.py", "--local"]]


def test_a_mode_takes_its_geometry_and_not_the_fill_the_route_sets():
    with pytest.raises(RecipeError, match="expected one of") as caught:
        build_object(
            {"type": "MarcatiliMode", "radius": "125 um", "gas": "He"},
            BuildContext(),
            "modes[0]",
        )
    assert caught.value.parameter == "modes[0]"
    for name in ("MarcatiliMode", "RectMode", "SlabMode", "ModeCollection.slab"):
        # A slab's fill is the run's on the planar route, and refused elsewhere.
        factory = REGISTRY[name]
        assert "pressure" in factory.omitted + factory.fill, name
    with pytest.raises(RecipeError, match="tableau"):
        build_object({"type": "RungeKutta", "tableau": "dopri5"}, BuildContext(), "x")


def test_every_object_form_names_registered_types():
    for route in ROUTES:
        for entry in parameters(route).values():
            for form in entry.forms:
                if form.kind == "object" and form.recipe:
                    assert set(form.types) <= set(REGISTRY), (entry.name, form.types)
                elif form.kind == "object":
                    # A form only Python holds names no type a recipe could.
                    assert set(form.types).isdisjoint(REGISTRY), (entry.name, form)


def test_a_rate_table_is_a_capillary_s_plasma():
    document = Recipe.example("rdw").to_dict()
    document["build"]["plasma"] = {
        "type": "OptmatRate",
        "source": "He",
        "wavelength": "800 nm",
    }
    keywords = Recipe.from_dict(document).keywords()
    assert type(keywords["plasma"]).__name__ == "OptmatRate"


def test_a_signal_stops_the_command_s_run_through_its_flag(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "soliton.toml"
    path.write_text(GNLSE)
    seen = {}

    def run(self, output, *, stop, **keywords):
        # Ctrl-C in-process; os.kill would terminate the process on Windows.
        signal.raise_signal(signal.SIGINT)
        seen["stopped"] = stop.is_set()
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(Recipe, "run", run)
    before = signal.getsignal(signal.SIGINT)
    status = main(["run", str(path), "-o", str(tmp_path / "out.h5")])
    assert status == 128 + signal.SIGINT
    assert seen == {"stopped": True}
    assert "solprop resume" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGINT) is before


def test_a_launch_table_builds_the_call_s_launch():
    document = Recipe.example("rdw").to_dict()
    launch = {"HE1,1": 0.9, "HE1,2": {"share": 0.1, "phase": 1.57}}
    document["build"]["launch"] = launch
    recipe = Recipe.from_dict(document)
    assert recipe.get('launch["HE1,2"].share') == 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = recipe.build()
        keywords: dict[str, Any] = recipe.keywords()
        direct = Simulation.from_capillary(**keywords)
        assert built.metadata().configuration_hash() == (
            direct.metadata().configuration_hash()
        )
    record = cast("dict[str, Any]", built.configuration["modal_launch"])
    assert record["pulses"][0]["shares"] == [0.9, 0.1]


# ------------------------------------------------- a canonical save is a fixed point
#
# Recipes are generated from the parameter metadata itself: a route, some of its
# keywords, and for each a value of a form the metadata declares, spelled as a recipe
# author might (a bare number in the keyword's unit, or a quantity string in that unit
# or in its display unit). Objects, mappings and callables are left to the examples:
# their tables are written as they stand.

PLAIN_KINDS = frozenset({"integer", "boolean", "choice", "string"})
LIST_KINDS = frozenset({"pair", "array"})
WRITTEN_KINDS = PLAIN_KINDS | LIST_KINDS | {"number"}


def _writable_form(parameter):
    """Return the first form of a keyword this generator writes, or ``None``."""
    return next(
        (
            form
            for form in parameter.forms
            if form.recipe and form.kind in WRITTEN_KINDS and form.elements is None
        ),
        None,
    )


def _numbers(form):
    """Return a strategy of finite numbers inside the form's range."""
    return st.floats(
        min_value=form.minimum,
        max_value=form.maximum,
        exclude_min=form.exclusive and form.minimum is not None,
        exclude_max=form.exclusive and form.maximum is not None,
        allow_nan=False,
        allow_infinity=False,
    )


def _spelled(number, form, parameter, spelling):
    """Write a number as a recipe may: bare, or a quantity in its unit or display."""
    if form.unit is None or spelling == "bare":
        return number
    unit = Unit(form.unit)
    display = parameter.display if spelling == "display" else form.unit
    return format_quantity(number, unit, display or form.unit)


# Read once, outside the strategies: the first reading imports the routes' modules,
# which Hypothesis would see touching the global random state mid-draw.
ROUTE_TABLES = {name: parameters(name) for name in (*ROUTES, "execution")}


def _written_form(parameter):
    """Return the form a keyword's generated values take."""
    form = _writable_form(parameter)
    assert form is not None, parameter.name
    return form


def _plain_values(form):
    """Return a strategy of a value that takes no unit, for the non-numeric kinds."""
    if form.kind == "choice":
        return st.sampled_from(form.choices)
    low = -(2**63) if form.minimum is None else int(form.minimum)
    return {
        "integer": st.integers(min_value=low, max_value=2**63 - 1),
        "boolean": st.booleans(),
        "string": st.text(max_size=10),
    }[form.kind]


@st.composite
def keyword_values(draw, parameter):
    """Return one value of a keyword, of a form its metadata declares."""
    form = _written_form(parameter)
    if parameter.none and draw(st.booleans()):
        return "none"
    if form.kind in PLAIN_KINDS:
        return draw(_plain_values(form))
    spelling = draw(st.sampled_from(["bare", "unit", "display"]))
    if form.kind == "number":
        return _spelled(draw(_numbers(form)), form, parameter, spelling)
    size = form.length or draw(st.integers(min_value=1, max_value=4))
    numbers = draw(st.lists(_numbers(form), min_size=size, max_size=size))
    return [_spelled(x, form, parameter, spelling) for x in numbers]


@st.composite
def recipe_documents(draw):
    """Return a recipe document of a route, written from its parameter metadata."""
    route = draw(st.sampled_from(ROUTES))
    table = ROUTE_TABLES[route]
    names = draw(
        st.lists(
            st.sampled_from(sorted(n for n, p in table.items() if _writable_form(p))),
            unique=True,
            max_size=8,
        )
    )
    document: dict[str, Any] = {
        "recipe": {"format": FORMAT, "route": route},
        "build": {name: draw(keyword_values(table[name])) for name in names},
    }
    if draw(st.booleans()):
        document["recipe"]["title"] = draw(st.text(max_size=20))
    execution = ROUTE_TABLES["execution"]
    chosen = draw(st.lists(st.sampled_from(["fields", "resume_history"]), unique=True))
    if chosen:
        document["execution"] = {
            name: draw(keyword_values(execution[name])) for name in chosen
        }
    if names and draw(st.booleans()):
        # A scan axis keyed by a parameter path, which the writer must quote.
        path = f"{names[0]}[0]" if draw(st.booleans()) else names[0]
        document["scan"] = {path: draw(st.lists(st.floats(0.0, 1.0), min_size=1))}
    return document


def _read_value(parameter, value):
    """Return a written value as numbers: each quantity read in its keyword's unit."""
    form = _written_form(parameter)
    if isinstance(value, list):
        return [_read_value(parameter, item) for item in value]
    if form.unit is None or value == "none" or isinstance(value, bool):
        return value
    return parse_quantity(value, Unit(form.unit))


@settings(max_examples=60, deadline=None)
@given(document=recipe_documents())
def test_a_canonical_save_reads_back_equal_and_saves_to_the_same_text(document):
    recipe = Recipe.from_dict(document)
    text = recipe.to_text()
    again = Recipe.loads(text)
    # Structural (class P): the canonical text is the recipe's value, so reading it
    # gives an equal recipe, and saving that again writes the same text: a fixed
    # point, which holds only if every quantity reads back to the same double and
    # no table, key or value is dropped or reordered on the way.
    assert again == recipe
    assert Recipe.from_dict(again.to_dict()).to_text() == text
    assert parse_text(text) == again.to_dict()
    # Equality compares canonical spellings, which a lossy spelling would pass on
    # both sides; so each value written must also read back to the same numbers.
    table = ROUTE_TABLES[recipe.route]
    for name, value in recipe.to_dict()["build"].items():
        written = again.get(name)
        assert _read_value(table[name], written) == _read_value(table[name], value)

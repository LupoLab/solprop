# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recipes: read, check, build, write and export (SEP 41, S1 and S10)."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import warnings
from typing import Any, cast

import numpy as np
import pytest

from solprop.cli import main
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.pulses import DataPulse, SechPulse
from solprop.recipe import Recipe, RecipeError, RecipeIssue, parameters
from solprop.recipe.registry import (
    REGISTRY,
    BuildContext,
    build_object,
    decode,
    decode_keyword,
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


def test_unknown_keys_and_routes_are_refused():
    with pytest.raises(RecipeError, match="wavelengthx"):
        Recipe.loads(GNLSE.replace("wavelength =", "wavelengthx ="))
    with pytest.raises(RecipeError, match="radial"):
        Recipe.loads(GNLSE.replace('route = "gnlse"', 'route = "radial"'))
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


def test_execution_settings_reach_the_settings_and_the_run(recipe):
    text = GNLSE + '\n[execution]\nfields = "planes"\nfield_planes = [0, -1]\n'
    simulation = Recipe.loads(text).build()
    assert simulation.settings.fields == "planes"
    assert simulation.settings.field_planes == (0, -1)
    for source, built in (
        (Recipe.loads(text + 'memory_budget = "none"\n'), simulation),
        (recipe, recipe.build()),
    ):
        metadata = source.run().metadata
        assert metadata is not None
        assert metadata.configuration_hash() == _hash(built)


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


def test_check_reports_each_value_at_its_path(recipe):
    bad = Recipe.loads(GNLSE.replace('length = "1 cm"', 'length = "1 bar"'))
    issues = bad.check()
    assert [issue.parameter for issue in issues] == ["length"]
    assert issues[0].severity == "error"
    missing = Recipe.loads(GNLSE.replace("gamma = 0.11\n", ""))
    assert RecipeIssue("gamma is required", "gamma", "error") in missing.check()
    assert recipe.check() == []


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
        assert "pressure" in REGISTRY[name].omitted, name
    with pytest.raises(RecipeError, match="tableau"):
        build_object({"type": "RungeKutta", "tableau": "dopri5"}, BuildContext(), "x")


def test_every_object_form_names_registered_types():
    for route in ("gnlse", "capillary", "waveguide"):
        for entry in parameters(route).values():
            for form in entry.forms:
                if form.kind == "object" and form.recipe:
                    assert set(form.types) <= set(REGISTRY), (entry.name, form.types)


def test_a_signal_stops_the_command_s_run_through_its_flag(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "soliton.toml"
    path.write_text(GNLSE)
    seen = {}

    def run(self, output, *, stop, **keywords):
        os.kill(os.getpid(), signal.SIGINT)
        seen["stopped"] = stop.is_set()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)

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

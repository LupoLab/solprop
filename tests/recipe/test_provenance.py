# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The recipe a run records, and the resolved recipe that rebuilds it (SEP 41, S2)."""

import dataclasses
import hashlib
import threading
import warnings

import numpy as np
import pytest

from solprop.driver.simulation import SimulationSettings
from solprop.exceptions import MaterialRangeWarning, SolpropWarning
from solprop.log import NullReporter
from solprop.output import MemoryOutput
from solprop.output.hdf5 import open_file
from solprop.physdata.mixtures import Mixture
from solprop.recipe import Recipe, parameters, provenance, resume_file
from solprop.recipe.document import parse_text
from solprop.recipe.registry import BuildContext, build_object
from solprop.stats import ApertureSet, StatsSpec

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

# The fast examples, one per route, and a capillary with its plasma and Raman rules.
EXAMPLES = ("soliton", "overview", "rdw")


def _hash(simulation):
    return simulation.metadata().configuration_hash()


def _resolved(recipe):
    simulation = recipe.build()
    simulation.translate()
    return simulation, provenance.resolved_document(recipe, simulation)


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_resolved_recipe_rebuilds_the_run(name):
    recipe = Recipe.example(name)
    simulation, document = _resolved(recipe)
    assert _hash(Recipe.from_dict(document).build()) == _hash(simulation)


@pytest.mark.parametrize("name", EXAMPLES)
def test_each_resolvable_keyword_written_back_keeps_the_hash(name):
    recipe = Recipe.example(name)
    simulation, document = _resolved(recipe)
    authored = recipe.to_dict()["build"]
    written = [key for key in document["build"] if key not in authored]
    table = parameters(recipe.route)
    assert written, "the example leaves resolvable keywords to the build"
    for key in written:
        if key == "seed":
            continue
        assert table[key].resolvable, key
        one = recipe.to_dict()
        one["build"][key] = document["build"][key]
        if key == "save_planes":
            one["build"].pop("n_save", None)
        assert _hash(Recipe.from_dict(one).build()) == _hash(simulation), key


def _soliton_with_execution(table):
    document = Recipe.example("soliton").to_dict()
    document["execution"] = table
    return Recipe.from_dict(document)


@pytest.mark.parametrize(
    ("name", "path", "value"),
    [
        # An integrator other than its type's default stays as authored, and the
        # route's default, an object, is not written in its place.
        ("soliton", "integrator", {"type": "Multistep", "order": 4}),
        # A temperature profile has no single temperature to write back.
        (
            "overview",
            "temperature",
            {
                "type": "PiecewiseLinear",
                "positions": ["0 m", "0.02 m"],
                "values": [290.0, 300.0],
            },
        ),
    ],
)
def test_an_authored_object_is_kept_and_rebuilds_the_run(name, path, value):
    recipe = Recipe.example(name)
    recipe.set(path, value)
    simulation, document = _resolved(recipe)
    assert document["build"][path] == value
    assert _hash(Recipe.from_dict(document).build()) == _hash(simulation)


def test_held_plane_goes_into_the_execution_table_only_when_turned_on():
    _, default = _resolved(Recipe.example("soliton"))
    assert "held_plane" not in default["execution"]
    simulation, document = _resolved(_soliton_with_execution({"held_plane": True}))
    assert document["execution"]["held_plane"] is True
    rebuilt = Recipe.from_dict(document).build()
    assert rebuilt.settings.held_plane is True
    # Outside the resume key: the record does not move.
    assert _hash(rebuilt) == _hash(Recipe.example("soliton").build())
    assert _hash(simulation) == _hash(rebuilt)


def test_resume_history_is_resolved_into_the_execution_table():
    recipe = _soliton_with_execution({"resume_history": True})
    simulation, document = _resolved(recipe)
    assert document["execution"]["resume_history"] is True
    rebuilt = Recipe.from_dict(document).build()
    assert rebuilt.settings.resume_history is True
    assert _hash(rebuilt) == _hash(simulation)


def test_a_statistics_selection_is_written_as_the_table_that_builds_it():
    spec = StatsSpec(
        names=("energy", "energy_window", "region_spectrum"),
        energy_windows=np.array([[1.0e15, 2.0e15]]),
        periodic=("energy",),
        period=3,
        spectral_regions=np.array([[-1.0, 1.0, -1.0, 1.0]]),
    )
    table = provenance._stats_table(spec)
    assert table is not None
    built = build_object(table, BuildContext(), "stats")
    assert isinstance(built, StatsSpec)
    assert (built.names, built.periodic, built.period) == (
        spec.names,
        spec.periodic,
        spec.period,
    )
    np.testing.assert_array_equal(built.energy_windows, spec.energy_windows)
    np.testing.assert_array_equal(built.spectral_regions, spec.spectral_regions)
    # Far-field apertures have no table, so such a selection is not written back.
    apertures = ApertureSet(centres=[[0.0, 0.0]], diameters=[1e-3], distances=[1.0])
    assert provenance._stats_table(StatsSpec(("energy",), apertures=apertures)) is None


def test_settings_without_a_controller_or_statistics_leave_them_to_the_rules():
    # Settings made without a route carry neither; every recipe route sets both.
    recipe = Recipe.example("soliton")
    simulation = recipe.build()
    settings = simulation.settings
    simulation._settings = SimulationSettings(
        save_planes=settings.save_planes,
        dz_max=settings.dz_max,
        dz_init=settings.dz_init,
    )
    assert simulation.settings.controller is None
    assert simulation.settings.stats is None
    values = provenance._used(recipe, simulation)
    table = parameters("gnlse")
    rule = table["rtol"].resolver
    assert rule is not None
    assert values["rtol"] == rule(recipe.to_dict()["build"]).value
    assert values["atol"] == table["atol"].default
    assert "stats" not in values


def test_the_keywords_left_to_the_build_would_move_the_record():
    # An explicit first step turns the start-up rule off (SEP 41 section 5.2).
    recipe = Recipe.example("soliton")
    simulation = recipe.build()
    length = simulation.settings.save_planes[-1]
    with_step = recipe.to_dict()
    with_step["build"]["dz_init"] = float(length * 1e-4)
    assert _hash(Recipe.from_dict(with_step).build()) != _hash(simulation)
    assert not parameters("gnlse")["dz_init"].resolvable
    # The fraction a Raman default takes records where it came from.
    raman = recipe.to_dict()
    raman["build"]["raman"] = True
    resolved = Recipe.from_dict(raman).build()
    fraction = dict(raman, build={**raman["build"], "raman_fraction": 0.18})
    assert _hash(Recipe.from_dict(fraction).build()) != _hash(resolved)
    for route in ("gnlse", "capillary"):
        table = parameters(route)
        for key in ("raman_fraction", "absorber", "backend", "grid"):
            entry = table.get(key)
            assert entry is None or not entry.resolvable


def test_an_unseeded_draw_is_resolved_with_its_seed():
    recipe = Recipe.example("soliton")
    recipe.set("noise", "input")
    simulation, document = _resolved(recipe)
    assert simulation.noise is not None
    assert int(str(document["build"]["seed"])) == simulation.noise.seed
    assert _hash(Recipe.from_dict(document).build()) == _hash(simulation)


def test_a_run_records_both_texts_once_and_a_resume_leaves_them(tmp_path):
    recipe = Recipe.example("soliton")
    path = tmp_path / "soliton.h5"
    flag = threading.Event()

    class Stop(NullReporter):
        def plane(self, z, summary):
            flag.set()

    recipe.run(path, progress=Stop(), stop=flag)
    with open_file(path) as handler:
        first = handler.recipe
        metadata = handler.metadata
        assert metadata is not None
        stamp = metadata.configuration_hash()
    assert first is not None
    assert first.authored == recipe.to_text()
    assert _hash(Recipe.loads(first.resolved).build()) == stamp
    # A later session, here from a recipe edited outside the resume key, is its own
    # record's business: the first recipe stays.
    edited = _soliton_with_execution({"held_plane": True})
    assert edited.to_text() != recipe.to_text()
    edited.run(path, resume=True, progress="none")
    with open_file(path) as handler:
        assert handler.recipe == first
        assert handler.outcome is not None and handler.outcome.status == "complete"


def test_the_execution_settings_are_written_as_at_run_start():
    recipe = Recipe.example("soliton")
    simulation = recipe.build()
    simulation.settings = dataclasses.replace(
        simulation.settings, fields="planes", field_planes=(0, -1)
    )
    output = MemoryOutput()
    simulation.run(output, progress="none")
    assert output.recipe is not None
    execution = parse_text(output.recipe.resolved)["execution"]
    assert execution["fields"] == "planes"
    assert execution["field_planes"] == [0, -1]


def test_a_change_the_recipe_does_not_describe_drops_it_and_warns():
    recipe = Recipe.example("soliton")
    simulation = recipe.build()
    with pytest.warns(SolpropWarning, match="dz_max changed"):
        simulation.settings = dataclasses.replace(simulation.settings, dz_max=1e-3)
    output = MemoryOutput()
    simulation.run(output, progress="none")
    assert output.recipe is None
    built = recipe.build()
    with pytest.warns(SolpropWarning, match="with_input derives"):
        derived = built.with_input(built._field0)
    output = MemoryOutput()
    derived.run(output, progress="none")
    assert output.recipe is None


def test_with_mixture_drops_the_recipe():
    recipe = Recipe.example("overview")
    simulation = recipe.build()
    with (
        warnings.catch_warnings(),
        pytest.warns(SolpropWarning, match="with_mixture derives"),
    ):
        warnings.simplefilter("ignore", MaterialRangeWarning)
        derived = simulation.with_mixture(Mixture.fill({"He": 1.0}, 0.5))
    assert derived._recipe is None


def _spectrum(tmp_path, size=200):
    # About the example's 835 nm carrier, 359 THz.
    frequency = np.linspace(315e12, 405e12, size)
    power = np.exp(-(((frequency - 359e12) / 10e12) ** 2))
    path = tmp_path / "spectrum.csv"
    np.savetxt(path, np.column_stack([frequency, power]), delimiter=",")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _measured(tmp_path):
    path, digest = _spectrum(tmp_path)
    text = Recipe.example("soliton").to_text().replace('width = "100 fs"', "")
    document = parse_text(text)
    build = document["build"]
    for key in ("width", "peak_power", "shape"):
        build.pop(key, None)
    build["pulses"] = [
        {
            "type": "load_spectrum",
            "file": path.name,
            "sha256": digest,
            "energy": "1 pJ",
        }
    ]
    return Recipe(document, base=tmp_path), path


def test_data_files_are_embedded_up_to_the_cap(tmp_path, monkeypatch):
    recipe, path = _measured(tmp_path)
    simulation = recipe.build()
    record = recipe.record(simulation)
    (entry,) = record.files.values()
    assert entry.path == path.name
    assert entry.data == path.read_bytes()
    monkeypatch.setattr(provenance, "EMBED_LIMIT", 10)
    (kept,) = recipe.record(simulation).files.values()
    assert kept.data is None
    assert kept.sha256 == entry.sha256


def test_a_file_written_from_a_script_has_no_recipe(tmp_path):
    simulation = Recipe.example("soliton").build()
    simulation._recipe = None
    simulation.run(tmp_path / "script.h5", progress="none")
    with open_file(tmp_path / "script.h5") as handler:
        assert handler.recipe is None


def test_an_embedded_data_file_rebuilds_the_run_without_the_original(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    recipe, spectrum = _measured(source)
    path = tmp_path / "measured.h5"
    flag = threading.Event()

    class Stop(NullReporter):
        def plane(self, z, summary):
            flag.set()

    recipe.run(path, progress=Stop(), stop=flag)
    stamp = _hash(recipe.build())
    spectrum.unlink()
    # The run continues from its file alone, the spectrum read from the output.
    resume_file(path, progress="none")
    with open_file(path) as handler:
        assert handler.outcome is not None and handler.outcome.status == "complete"
    # Saved elsewhere, the recipe writes the embedded file beside it under its name.
    copy = tmp_path / "copy" / "measured.toml"
    copy.parent.mkdir()
    Recipe.from_output(path).save(copy)
    (written,) = (entry for entry in copy.parent.iterdir() if entry != copy)
    assert written.name == "file-0.csv"
    assert 'file = "file-0.csv"' in copy.read_text()
    assert _hash(Recipe.load(copy).build()) == stamp


def test_a_data_file_kept_by_reference_stays_a_reference(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    recipe, _ = _measured(source)
    monkeypatch.setattr(provenance, "EMBED_LIMIT", 10)
    path = tmp_path / "referred.h5"
    flag = threading.Event()

    class Stop(NullReporter):
        def plane(self, z, summary):
            flag.set()

    recipe.run(path, progress=Stop(), stop=flag)
    copy = tmp_path / "copy" / "referred.toml"
    copy.parent.mkdir()
    Recipe.from_output(path).save(copy)
    assert list(copy.parent.iterdir()) == [copy]
    assert 'file = "spectrum.csv"' in copy.read_text()
    (issue,) = Recipe.load(copy).check()
    assert issue.severity == "error"
    assert issue.parameter == "pulses[0]"

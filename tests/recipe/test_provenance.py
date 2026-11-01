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

from solprop.exceptions import MaterialRangeWarning, SolpropWarning
from solprop.log import NullReporter
from solprop.output import MemoryOutput
from solprop.output.hdf5 import open_file
from solprop.physdata.mixtures import Mixture
from solprop.recipe import Recipe, parameters, provenance
from solprop.recipe.document import parse_text

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
    # A later session, here from an edited recipe that changes only the numerics, is
    # its own record's business: the first recipe stays.
    edited = Recipe.example("soliton")
    edited.set("rtol", 1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
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

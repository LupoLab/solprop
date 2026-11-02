# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The editable recipe and the field values."""

from pathlib import Path

import pytest
from solprop.recipe.errors import RecipeError

from conftest import TINY
from solprop_gui.model.recipe import MISSING, RecipeModel, path_text
from solprop_gui.model.values import FieldError, inline, read_value

COMMENTED = TINY.replace("[build]", "# a comment the file keeps\n[build]")


def test_an_unedited_recipe_saves_verbatim(tmp_path: Path):
    model = RecipeModel(COMMENTED)
    assert not model.edited and not model.dirty
    assert model.route == "capillary" and model.title == "Tiny capillary"
    target = model.save(tmp_path / "a.toml")
    assert target.read_text(encoding="utf-8") == COMMENTED
    assert RecipeModel.load(target).path == target


def test_an_edit_raises_the_revision_and_writes_canonically(tmp_path: Path):
    model = RecipeModel(COMMENTED, path=tmp_path / "a.toml")
    assert model.set("energy", "150 uJ") == "100 uJ"
    assert model.revision == 2 and model.dirty and model.edited
    assert "a comment" not in model.to_text()
    assert 'energy = "150 uJ"' in model.to_text()
    model.save()
    assert not model.dirty
    assert "150 uJ" in (tmp_path / "a.toml").read_text(encoding="utf-8")


def test_unset_removes_a_value_and_undo_puts_it_back():
    model = RecipeModel(TINY)
    previous = model.set("pressure", MISSING)
    assert previous == "1 bar" and model.get("pressure") is MISSING
    assert model.unset("pressure") is MISSING
    model.set("pressure", previous)
    assert model.get("pressure") == "1 bar"


def test_nested_values_and_other_tables():
    model = RecipeModel(TINY)
    model.set("stats", {"type": "StatsSpec", "names": ["energy", "peak_power"]})
    assert model.get("stats.names[1]") == "peak_power"
    assert model.unset("stats.type") == "StatsSpec"
    assert model.get("stats") == {"names": ["energy", "peak_power"]}
    with pytest.raises(RecipeError, match="array element"):
        model.unset("stats.names[0]")
    model.set("fields", "planes", table="execution")
    assert model.get("fields", table="execution") == "planes"
    model.unset("fields", table="execution")
    assert "execution" not in model.document
    assert model.get("anything", table="pins") is MISSING


def test_the_title_can_be_set_and_removed():
    model = RecipeModel(TINY)
    assert model.set_title("New") == "Tiny capillary"
    assert model.title == "New"
    model.set_title(None)
    assert model.title is None


def test_save_needs_a_file_and_recovery_leaves_the_state(tmp_path: Path):
    model = RecipeModel(TINY)
    with pytest.raises(ValueError, match="no file"):
        model.save()
    model.set("energy", "1 uJ")
    copy = model.write_recovery(tmp_path / "deep" / "recovery.toml")
    assert "1 uJ" in copy.read_text(encoding="utf-8")
    assert model.dirty


def test_a_document_that_is_not_a_recipe_is_refused():
    with pytest.raises(RecipeError):
        RecipeModel("[nonsense]\nx = 1\n")


def test_path_text_writes_keys_that_are_not_names():
    assert path_text("pulses", 1, "energy") == "pulses[1].energy"
    assert path_text("launch", "HE1,1:even", "share") == 'launch["HE1,1:even"].share'


# ---------------------------------------------------------------------- values


def test_inline_writes_toml():
    assert inline("1 bar") == "1 bar"
    assert inline(False) == "false"
    assert inline(3) == "3"
    assert inline(float("nan")) == "nan"
    assert inline(float("inf")) == "inf" and inline(float("-inf")) == "-inf"
    assert inline({}) == "{}"
    assert inline({"a b": 'q"'}) == '{ "a b" = "q\\"" }'
    assert inline([1, "x"]) == '[1, "x"]'
    assert inline(Path("p")) == '"p"'


NUMBER = [{"kind": "number", "unit": "J"}]


def test_read_value_checks_a_quantity_against_the_unit():
    assert read_value(" 275 uJ ", NUMBER) == "275 uJ"
    assert read_value("2e-4", NUMBER) == 2e-4
    with pytest.raises(FieldError, match="time"):
        read_value("10 fs", NUMBER)


def test_missing_says_what_it_is():
    assert repr(MISSING) == "MISSING"


def test_read_value_takes_the_other_forms():
    assert read_value("abc", [{"kind": "integer"}, {"kind": "string"}]) == "abc"
    assert read_value("3", [{"kind": "integer"}]) == 3
    assert read_value("[1, 2]", [{"kind": "integer"}, {"kind": "array"}]) == [1, 2]
    assert read_value("He", [{"kind": "string"}]) == "He"
    assert read_value("{ He = 0.5 }", [{"kind": "mapping"}]) == {"He": 0.5}
    assert read_value("1e-6", [{"kind": "number", "unit": None}]) == 1e-6
    pair = [{"kind": "pair", "unit": "m"}]
    assert read_value('["200 nm", "2 um"]', pair) == ["200 nm", "2 um"]
    with pytest.raises(FieldError, match="not a value"):
        read_value("[1,", pair)
    with pytest.raises(FieldError):
        read_value("abc", [{"kind": "number", "unit": None}])


# ---------------------------------------------------------------------- quantities


def test_a_quantity_is_split_converted_and_written():
    from solprop_gui.model.quantities import (  # noqa: PLC0415
        number_in,
        split,
        unit_of,
        write,
    )

    assert split("1.5e3 fs") == ("1.5e3", "fs")
    assert split("nonsense 3") == ("", "")
    assert unit_of("2 um", ("nm", "um"), "nm") == "um"
    assert unit_of("2 pm", ("nm", "um"), "nm") == "nm"
    assert unit_of(2e-6, ("nm",), "nm") == "nm"
    assert number_in("1.2 bar", "bar", "mbar") == "1200"
    assert number_in(2e-6, "m", "um") == "2"
    assert write("3", "J", "uJ") == "3 uJ"
    with pytest.raises(RecipeError):
        write("3 fs", "J", "uJ")
    with pytest.raises(RecipeError, match="not a number"):
        write("uJ", "J", "uJ")


def test_units_on_offer():
    from solprop_gui.registry.units import pretty, unit_choices  # noqa: PLC0415

    assert unit_choices("bar", "mbar") == ("mbar", "bar", "kPa", "Pa")
    assert unit_choices("m", "km") == ("nm", "um", "mm", "cm", "m", "km")
    assert unit_choices("rad") == ("rad",)
    assert pretty("um") == "µm" and pretty("m^-1") == "m⁻¹" and pretty("W") == "W"

# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A recipe's [scan] table, run through run_scan (SEP 41, S1)."""

import pickle

import numpy as np
import pytest

from solprop.driver.scan import Scan, run_scan
from solprop.recipe import Recipe, RecipeError
from solprop.recipe.document import parse_text

TEXT = """\
[recipe]
format = 1
route = "gnlse"

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
backend = "numpy_ref"

[execution]
fields = "planes"
field_planes = [-1]
"""


def _with_scan(table):
    document = parse_text(TEXT)
    document["scan"] = table
    return Recipe.from_dict(document)


def test_the_scan_reads_each_axis_in_its_keyword_s_unit():
    recipe = _with_scan(
        {"name": "powers", "peak_power": ["5 W", 10], "width": ["50 fs", "0.1 ps"]}
    )
    scan = recipe.scan()
    assert isinstance(scan.scan, Scan)
    assert scan.scan.shape == (2, 2)
    assert scan.scan.variables["peak_power"].tolist() == [5.0, 10.0]
    assert scan.scan.variables["width"].tolist() == [50e-15, 100e-15]
    assert scan.keywords == {"fields": "planes", "field_planes": (-1,)}


@pytest.mark.parametrize(
    ("spacing", "expected"),
    [("linear", [1.0, 5.5, 10.0]), ("log", [1.0, np.sqrt(10.0), 10.0])],
)
def test_a_range_is_spaced_as_asked(spacing, expected):
    scan = _with_scan(
        {
            "name": "range",
            "peak_power": {
                "start": "1 W",
                "stop": "10 W",
                "num": 3,
                "spacing": spacing,
            },
        }
    ).scan()
    assert np.allclose(scan.scan.variables["peak_power"], expected, rtol=1e-15)


def test_a_scan_without_a_name_or_table_or_with_a_wrong_path_is_refused():
    with pytest.raises(RecipeError, match="no \\[scan\\] table"):
        Recipe.loads(TEXT).scan()
    with pytest.raises(RecipeError, match="needs a name"):
        _with_scan({"peak_power": [1.0]}).scan()
    with pytest.raises(RecipeError, match="peak_powr"):
        _with_scan({"name": "x", "peak_powr": [1.0]}).scan()
    with pytest.raises(RecipeError, match="spacing"):
        _with_scan(
            {
                "name": "x",
                "peak_power": {"start": 1, "stop": 2, "num": 2, "spacing": "x"},
            }
        ).scan()
    with pytest.raises(RecipeError, match="a list of values"):
        _with_scan({"name": "x", "peak_power": 1.0}).scan()


def test_the_builder_pickles_and_passes_the_point_s_seed():
    recipe = _with_scan({"name": "noise", "peak_power": [5.0, 10.0], "base_seed": 7})
    document = recipe.to_dict()
    document["build"]["noise"] = "input"
    scan = Recipe.from_dict(document).scan()
    builder = pickle.loads(pickle.dumps(scan.build))
    point = scan.scan.point(1)
    simulation = builder(point)
    assert simulation.noise is not None
    assert simulation.noise.seed == point.seed
    assert simulation.settings.fields == "all"  # run_scan takes [execution] itself


def test_a_recipe_s_scan_runs(tmp_path):
    scan = _with_scan({"name": "run", "peak_power": ["5 W", "10 W"]}).scan()
    output = run_scan(
        scan.scan,
        scan.build,
        output=tmp_path / "scan.h5",
        progress="none",
        **scan.keywords,  # type: ignore[arg-type]
    )
    assert [str(status) for status in output.status] == ["done", "done"]
